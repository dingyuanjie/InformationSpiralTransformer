"""Level 7.6.1: high-powered paired evaluation of frozen Level 7.6 checkpoints."""
import argparse, json, math, time
from pathlib import Path
import torch
import torch.nn.functional as F
from marked_retrieval_level2 import make_batch
from long_context_test import set_seed
from run_level7_6_local import build, matched_width, params, LONG_LENGTHS, SEEDS

ROOT=Path(__file__).resolve().parent
PARENT=ROOT/'experiments/level7_6/formal'
VARIANTS=('transformer-matched','ist-full','ist-stable')

def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2),encoding='utf-8')

def wilson(correct,total,z=1.959963984540054):
    p=correct/total; den=1+z*z/total; center=(p+z*z/(2*total))/den
    half=z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/den
    return [center-half,center+half]

@torch.no_grad()
def evaluate(model,length,samples,seed,device,dtype):
    set_seed(seed); model.eval(); correctness=[]
    if device.type=='cuda': torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    started=time.perf_counter()
    for _ in range(samples):
        x,y,_=make_batch(1,length,length-3,16,device)
        with torch.autocast(device_type='cuda',dtype=dtype): logits=model(x)[...,:16]
        correctness.append(int(logits[:,-1].argmax(-1).item()==y.item()))
    torch.cuda.synchronize(); seconds=time.perf_counter()-started
    correct=sum(correctness)
    return {'length':length,'samples':samples,'correct':correct,'accuracy':correct/samples,
            'wilson95':wilson(correct,samples),'seconds':seconds,
            'tokens_per_second':samples*length/seconds,
            'peak_memory_mb':torch.cuda.max_memory_allocated()/1048576,
            'correctness':correctness}

def paired(a,b):
    delta=[x-y for x,y in zip(a,b)]; n=len(delta); mean=sum(delta)/n
    variance=sum((x-mean)**2 for x in delta)/(n-1); half=1.959963984540054*math.sqrt(variance/n)
    return {'difference':mean,'normal95':[mean-half,mean+half],
            'improved':sum(x==1 for x in delta),'harmed':sum(x==-1 for x in delta),'ties':sum(x==0 for x in delta)}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--samples',type=int,default=1000); p.add_argument('--output',default='experiments/level7_6_1/formal')
    p.add_argument('--dry-run',action='store_true'); p.add_argument('--force',action='store_true'); args=p.parse_args()
    if args.samples!=1000 and not args.dry_run: raise ValueError('Formal protocol locks samples=1000')
    if args.dry_run: print(json.dumps({'variants':VARIANTS,'seeds':SEEDS,'lengths':LONG_LENGTHS,'samples':args.samples},indent=2)); return
    if not torch.cuda.is_available(): raise RuntimeError('CUDA GPU required')
    device=torch.device('cuda'); dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    target=params(build('ist-full',96)); width,_=matched_width(target); root=ROOT/args.output; root.mkdir(parents=True,exist_ok=True)
    runs=[]
    for variant in VARIANTS:
        for seed in SEEDS:
            out=root/f'{variant}_seed{seed}.json'
            if out.exists() and not args.force: runs.append(json.loads(out.read_text())); continue
            checkpoint=PARENT/f'{variant}_seed{seed}'/'stage3.pt'
            if not checkpoint.exists(): raise FileNotFoundError(checkpoint)
            model=build(variant,width).to(device); state=torch.load(checkpoint,map_location=device,weights_only=False); model.load_state_dict(state['model'])
            tests=[]
            for length in LONG_LENGTHS:
                eval_seed=7610000+seed*10+LONG_LENGTHS.index(length)
                row=evaluate(model,length,args.samples,eval_seed,device,dtype); tests.append(row)
                print(f'{variant} seed={seed} length={length} accuracy={row["accuracy"]:.2%}',flush=True)
            run={'variant':variant,'seed':seed,'checkpoint':str(checkpoint.relative_to(ROOT)).replace('\\','/'),'tests':tests}
            save(out,run); runs.append(run); torch.cuda.empty_cache()
            save(root/'runs.partial.json',runs)
    keyed={(r['variant'],r['seed']):r for r in runs}; comparisons=[]
    for variant in ('ist-full','ist-stable'):
        for seed in SEEDS:
            for i,length in enumerate(LONG_LENGTHS):
                comparisons.append({'variant':variant,'seed':seed,'length':length,
                    **paired(keyed[(variant,seed)]['tests'][i]['correctness'],keyed[('transformer-matched',seed)]['tests'][i]['correctness'])})
    summary=[]
    for variant in VARIANTS:
        for i,length in enumerate(LONG_LENGTHS):
            correct=sum(keyed[(variant,s)]['tests'][i]['correct'] for s in SEEDS); total=args.samples*len(SEEDS)
            summary.append({'variant':variant,'length':length,'correct':correct,'samples':total,'accuracy':correct/total,'wilson95':wilson(correct,total)})
    result={'protocol':vars(args),'matched_width':width,'summary':summary,'paired_comparisons':comparisons,'runs':runs}
    save(root/'result.json',result); print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
