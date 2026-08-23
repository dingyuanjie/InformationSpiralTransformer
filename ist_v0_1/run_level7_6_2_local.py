"""Level 7.6.2: paired held-out position and repeated-evidence evaluation."""
import argparse, json, math, time
from pathlib import Path
import torch
from long_context_test import set_seed
from run_level7_6_local import build, matched_width, params, LONG_LENGTHS, SEEDS

ROOT=Path(__file__).resolve().parent; PARENT=ROOT/'experiments/level7_6/formal'
VARIANTS=('transformer-matched','ist-full','ist-stable')
TASKS=('early','middle','late','repeated4')

def save(path,value): path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(value,indent=2),encoding='utf-8')
def wilson(c,n,z=1.959963984540054):
    p=c/n; d=1+z*z/n; m=(p+z*z/(2*n))/d; h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d; return [m-h,m+h]

def make_heldout(length,task,device):
    vocab=16; mask,needle,query=vocab,vocab+1,vocab+2
    target=torch.randint(vocab,(1,),device=device); tokens=torch.randint(vocab,(1,length),device=device)
    if task=='repeated4': positions=[max(0,min(length-3,int(length*f))) for f in (.05,.30,.55,.80)]
    else:
        bands={'early':(0,.2),'middle':(.4,.6),'late':(.75,.95)}; lo,hi=bands[task]
        start=int(length*lo); stop=max(start+1,min(length-2,int(length*hi))); positions=[int(torch.randint(start,stop,(1,),device=device).item())]
    for pos in positions: tokens[0,pos]=needle; tokens[0,pos+1]=target
    tokens[0,-2]=query; tokens[0,-1]=mask
    return tokens,target

@torch.no_grad()
def evaluate(model,length,task,samples,seed,device,dtype):
    set_seed(seed); model.eval(); correctness=[]; torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); started=time.perf_counter()
    for _ in range(samples):
        x,y=make_heldout(length,task,device)
        with torch.autocast(device_type='cuda',dtype=dtype): logits=model(x)[...,:16]
        correctness.append(int(logits[:,-1].argmax(-1).item()==y.item()))
    torch.cuda.synchronize(); seconds=time.perf_counter()-started; correct=sum(correctness)
    return {'length':length,'task':task,'samples':samples,'correct':correct,'accuracy':correct/samples,'wilson95':wilson(correct,samples),'seconds':seconds,'tokens_per_second':samples*length/seconds,'peak_memory_mb':torch.cuda.max_memory_allocated()/1048576,'correctness':correctness}

def paired(a,b):
    delta=[x-y for x,y in zip(a,b)]; n=len(delta); mean=sum(delta)/n; var=sum((x-mean)**2 for x in delta)/(n-1); h=1.959963984540054*math.sqrt(var/n)
    return {'difference':mean,'normal95':[mean-h,mean+h],'improved':delta.count(1),'harmed':delta.count(-1),'ties':delta.count(0)}

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--samples-per-seed-task',type=int,default=250); p.add_argument('--output',default='experiments/level7_6_2/formal'); p.add_argument('--dry-run',action='store_true'); p.add_argument('--force',action='store_true'); args=p.parse_args()
    if args.samples_per_seed_task!=250 and not args.dry_run: raise ValueError('Formal protocol locks 250 samples per seed/task')
    protocol={'variants':VARIANTS,'seeds':SEEDS,'lengths':LONG_LENGTHS,'tasks':TASKS,'samples_per_seed_task':args.samples_per_seed_task}
    if args.dry_run: print(json.dumps(protocol,indent=2)); return
    if not torch.cuda.is_available(): raise RuntimeError('CUDA GPU required')
    device=torch.device('cuda'); dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16; target=params(build('ist-full',96)); width,_=matched_width(target); root=ROOT/args.output; root.mkdir(parents=True,exist_ok=True); runs=[]
    for variant in VARIANTS:
      for seed in SEEDS:
        out=root/f'{variant}_seed{seed}.json'
        if out.exists() and not args.force: runs.append(json.loads(out.read_text())); continue
        checkpoint=PARENT/f'{variant}_seed{seed}'/'stage3.pt'; model=build(variant,width).to(device); model.load_state_dict(torch.load(checkpoint,map_location=device,weights_only=False)['model']); tests=[]
        for li,length in enumerate(LONG_LENGTHS):
          for ti,task in enumerate(TASKS):
            eval_seed=7620000+seed*100+li*10+ti; row=evaluate(model,length,task,args.samples_per_seed_task,eval_seed,device,dtype); tests.append(row); print(f'{variant} seed={seed} L={length} task={task} acc={row["accuracy"]:.2%}',flush=True)
        run={'variant':variant,'seed':seed,'tests':tests}; save(out,run); runs.append(run); save(root/'runs.partial.json',runs); del model; torch.cuda.empty_cache()
    keyed={(r['variant'],r['seed']):r for r in runs}; summary=[]; comparisons=[]
    for li,length in enumerate(LONG_LENGTHS):
      for ti,task in enumerate(TASKS):
        idx=li*len(TASKS)+ti
        for variant in VARIANTS:
          rows=[keyed[(variant,s)]['tests'][idx] for s in SEEDS]; c=sum(x['correct'] for x in rows); n=sum(x['samples'] for x in rows); summary.append({'variant':variant,'length':length,'task':task,'correct':c,'samples':n,'accuracy':c/n,'wilson95':wilson(c,n)})
        for variant in ('ist-full','ist-stable'):
          a=[]; b=[]
          for s in SEEDS: a+=keyed[(variant,s)]['tests'][idx]['correctness']; b+=keyed[('transformer-matched',s)]['tests'][idx]['correctness']
          comparisons.append({'variant':variant,'length':length,'task':task,**paired(a,b)})
    save(root/'result.json',{'protocol':protocol,'matched_width':width,'summary':summary,'paired_comparisons':comparisons,'runs':runs}); print('level7_6_2_COMPLETE')
if __name__=='__main__': main()
