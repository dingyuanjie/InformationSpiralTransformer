"""Level 7.6.3: fixed-length exact Memory retention curve."""
import argparse,json,math,time
from pathlib import Path
import torch
from long_context_test import set_seed
from run_level7_6_local import build,matched_width,params,SEEDS

ROOT=Path(__file__).resolve().parent; PARENT=ROOT/'experiments/level7_6/formal'
VARIANTS=('transformer-matched','ist-full','ist-stable'); LENGTH=8192
DISTANCES=(32,64,128,256,512,1024,2048,4096,8190)

def save(path,value): path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(value,indent=2),encoding='utf-8')
def wilson(c,n,z=1.959963984540054):
 p=c/n; q=1+z*z/n; m=(p+z*z/(2*n))/q; h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/q; return [m-h,m+h]
def make_batch(distance,device):
 vocab=16; target=torch.randint(vocab,(1,),device=device); tokens=torch.randint(vocab,(1,LENGTH),device=device); pos=LENGTH-2-distance
 tokens[0,pos]=vocab+1; tokens[0,pos+1]=target; tokens[0,-2]=vocab+2; tokens[0,-1]=vocab; return tokens,target

@torch.no_grad()
def evaluate(model,distance,samples,seed,device,dtype):
 set_seed(seed); model.eval(); correctness=[]; torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); started=time.perf_counter()
 for _ in range(samples):
  x,y=make_batch(distance,device)
  with torch.autocast(device_type='cuda',dtype=dtype): logits=model(x)[...,:16]
  correctness.append(int(logits[:,-1].argmax(-1).item()==y.item()))
 torch.cuda.synchronize(); seconds=time.perf_counter()-started; c=sum(correctness)
 return {'distance':distance,'samples':samples,'correct':c,'accuracy':c/samples,'wilson95':wilson(c,samples),'seconds':seconds,'tokens_per_second':samples*LENGTH/seconds,'peak_memory_mb':torch.cuda.max_memory_allocated()/1048576,'correctness':correctness}

def paired(a,b):
 d=[x-y for x,y in zip(a,b)]; n=len(d); m=sum(d)/n; v=sum((x-m)**2 for x in d)/(n-1); h=1.959963984540054*math.sqrt(v/n)
 return {'difference':m,'normal95':[m-h,m+h],'improved':d.count(1),'harmed':d.count(-1),'ties':d.count(0)}

def main():
 p=argparse.ArgumentParser(description=__doc__); p.add_argument('--samples-per-seed-distance',type=int,default=200); p.add_argument('--output',default='experiments/level7_6_3/formal'); p.add_argument('--dry-run',action='store_true'); p.add_argument('--force',action='store_true'); args=p.parse_args()
 if args.samples_per_seed_distance!=200 and not args.dry_run: raise ValueError('Formal protocol locks 200 samples per seed/distance')
 protocol={'variants':VARIANTS,'seeds':SEEDS,'fixed_length':LENGTH,'distances':DISTANCES,'samples_per_seed_distance':args.samples_per_seed_distance}
 if args.dry_run: print(json.dumps(protocol,indent=2)); return
 if not torch.cuda.is_available(): raise RuntimeError('CUDA GPU required')
 device=torch.device('cuda'); dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16; target=params(build('ist-full',96)); width,_=matched_width(target); root=ROOT/args.output; root.mkdir(parents=True,exist_ok=True); runs=[]
 for variant in VARIANTS:
  for seed in SEEDS:
   out=root/f'{variant}_seed{seed}.json'
   if out.exists() and not args.force: runs.append(json.loads(out.read_text())); continue
   checkpoint=PARENT/f'{variant}_seed{seed}'/'stage3.pt'; model=build(variant,width).to(device); model.load_state_dict(torch.load(checkpoint,map_location=device,weights_only=False)['model']); tests=[]
   for i,distance in enumerate(DISTANCES):
    row=evaluate(model,distance,args.samples_per_seed_distance,7630000+seed*20+i,device,dtype); tests.append(row); print(f'{variant} seed={seed} distance={distance} accuracy={row["accuracy"]:.2%}',flush=True)
   run={'variant':variant,'seed':seed,'tests':tests}; save(out,run); runs.append(run); save(root/'runs.partial.json',runs); del model; torch.cuda.empty_cache()
 keyed={(r['variant'],r['seed']):r for r in runs}; summary=[]; comparisons=[]
 for i,distance in enumerate(DISTANCES):
  for variant in VARIANTS:
   rows=[keyed[(variant,s)]['tests'][i] for s in SEEDS]; c=sum(x['correct'] for x in rows); n=sum(x['samples'] for x in rows); ci=wilson(c,n); summary.append({'variant':variant,'distance':distance,'correct':c,'samples':n,'accuracy':c/n,'wilson95':ci,'above_random':ci[0]>1/16})
  for variant in ('ist-full','ist-stable'):
   a=[]; b=[]
   for s in SEEDS: a+=keyed[(variant,s)]['tests'][i]['correctness']; b+=keyed[('transformer-matched',s)]['tests'][i]['correctness']
   comparisons.append({'variant':variant,'distance':distance,**paired(a,b)})
 effective={v:max([x['distance'] for x in summary if x['variant']==v and x['above_random']],default=None) for v in VARIANTS}
 save(root/'result.json',{'protocol':protocol,'matched_width':width,'summary':summary,'paired_comparisons':comparisons,'effective_distance_wilson_lower_above_random':effective,'runs':runs}); print(json.dumps(effective,indent=2))
if __name__=='__main__': main()
