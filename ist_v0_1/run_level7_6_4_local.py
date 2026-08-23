"""Level 7.6.4: controlled long-length curriculum and frozen 8192 evaluation."""
from __future__ import annotations
import argparse,json,time
from pathlib import Path
import torch
import torch.nn.functional as F
from long_context_test import set_seed
from marked_retrieval_level2 import make_batch
from run_level7_6_local import build,matched_width,params,SEEDS
from run_level7_4_1_local import atomic_torch_save
from run_level7_1_local import atomic_save

ROOT=Path(__file__).resolve().parent
PARENT=ROOT/'experiments/level7_6/formal'
VARIANTS=('transformer-matched','ist-full','ist-stable')
STAGES=((1024,400,4),(2048,300,2),(4096,150,1))
EVAL_LENGTH=8192
WINDOWS=((16,63),(64,127),(128,255),(256,511),(512,1023),(1024,2047),(2048,4095),(4096,6143),(6144,8190))

def checkpoint_path(folder:Path,length:int)->Path: return folder/f'stage_{length}.pt'
def resume_path(folder:Path,length:int)->Path: return folder/f'stage_{length}_resume.pt'

def load_rng(state):
    torch.set_rng_state(state['torch_rng_state'].cpu())
    torch.cuda.set_rng_state(state['cuda_rng_state'].cpu())

def save_state(path,model,optimizer,length,step,history):
    atomic_torch_save(path,{'model':model.state_dict(),'optimizer':optimizer.state_dict(),'length':length,'stage_step':step,'history':history,'torch_rng_state':torch.get_rng_state(),'cuda_rng_state':torch.cuda.get_rng_state()})

@torch.no_grad()
def evaluate_window(model,window,samples,seed,device,dtype):
    set_seed(seed); model.eval(); correct=[]; torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); started=time.perf_counter()
    for _ in range(samples):
        target=torch.randint(16,(1,),device=device); tokens=torch.randint(16,(1,EVAL_LENGTH),device=device)
        distance=int(torch.randint(window[0],window[1]+1,(1,),device=device).item()); pos=EVAL_LENGTH-2-distance
        tokens[0,pos]=17; tokens[0,pos+1]=target; tokens[0,-2]=18; tokens[0,-1]=16
        with torch.autocast(device_type='cuda',dtype=dtype): logits=model(tokens)[...,:16]
        correct.append(int(logits[:,-1].argmax(-1).item()==target.item()))
    torch.cuda.synchronize(); seconds=time.perf_counter()-started
    return {'window':list(window),'samples':samples,'correct':sum(correct),'accuracy':sum(correct)/samples,'correctness':correct,'seconds':seconds,'tokens_per_second':samples*EVAL_LENGTH/seconds,'peak_memory_mb':torch.cuda.max_memory_allocated()/1048576}

def train_stage(model,optimizer,variant,seed,length,steps,batch,folder,args,device,dtype):
    final=checkpoint_path(folder,length); resume=resume_path(folder,length)
    if final.exists() and not args.force:
        state=torch.load(final,map_location=device,weights_only=False); model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer']); load_rng(state); return {'length':length,'status':'complete','steps':steps,'history':state['history']}
    history=[]; start=0
    resume_candidates=[p for p in (resume,resume.with_suffix(resume.suffix+'.tmp')) if p.exists()]
    if resume_candidates and not args.force:
        loaded=[(torch.load(p,map_location=device,weights_only=False),p) for p in resume_candidates]
        state,selected=max(loaded,key=lambda item:int(item[0]['stage_step']))
        model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer']); load_rng(state); start=int(state['stage_step']); history=state['history']; print(f'{variant} seed={seed} length={length} resume={start} source={selected.name}',flush=True)
    else: set_seed(7640000+seed+length)
    torch.cuda.reset_peak_memory_stats(); started=time.perf_counter()
    try:
        for step in range(start+1,steps+1):
            model.train(); x,y,pos=make_batch(batch,length,length-3,16,device); optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda',dtype=dtype):
                logits=model(x)[...,:16]; rows=torch.arange(batch,device=device); q=F.cross_entropy(logits[:,-1],y); local=F.cross_entropy(logits[rows,pos],y); loss=q+0.5*local+(0.1*model.memory_diversity_loss() if variant.startswith('ist-') else 0)
            loss.backward(); torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.0); optimizer.step()
            if step==1 or step%50==0:
                row={'step':step,'loss':float(loss.detach()),'query_loss':float(q.detach()),'local_loss':float(local.detach())}; history.append(row); save_state(resume,model,optimizer,length,step,history); print(f'{variant} seed={seed} length={length} step={step} loss={row["loss"]:.4f}',flush=True)
        save_state(final,model,optimizer,length,steps,history)
        return {'length':length,'status':'complete','steps':steps,'seconds':time.perf_counter()-started,'peak_memory_mb':torch.cuda.max_memory_allocated()/1048576,'history':history}
    except torch.OutOfMemoryError as exc:
        torch.cuda.empty_cache(); return {'length':length,'status':'oom','completed_step':history[-1]['step'] if history else start,'error':str(exc)}

def run_one(variant,seed,args,device,dtype,width,root):
    folder=root/f'{variant}_seed{seed}'; folder.mkdir(parents=True,exist_ok=True); result_path=folder/'result.json'
    if result_path.exists() and not args.force: return json.loads(result_path.read_text())
    model=build(variant,width).to(device); optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3)
    parent=torch.load(PARENT/f'{variant}_seed{seed}'/'stage3.pt',map_location=device,weights_only=False); model.load_state_dict(parent['model']); optimizer.load_state_dict(parent['optimizer']); load_rng(parent)
    stages=[]
    for length,steps,batch in STAGES:
        row=train_stage(model,optimizer,variant,seed,length,steps,batch,folder,args,device,dtype); stages.append(row)
        if row['status']=='oom': break
    trained_through=max([x['length'] for x in stages if x['status']=='complete'],default=512); tests=[]
    for i,window in enumerate(WINDOWS):
        try: tests.append(evaluate_window(model,window,args.eval_samples,7650000+seed*20+i,device,dtype))
        except torch.OutOfMemoryError as exc: torch.cuda.empty_cache(); tests.append({'window':list(window),'status':'oom','error':str(exc)})
    result={'variant':variant,'seed':seed,'trained_through':trained_through,'l3_slot_frozen':variant=='ist-stable','stages':stages,'tests_8192':tests}; atomic_save(result_path,result); return result

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--eval-samples',type=int,default=50); p.add_argument('--output',default='experiments/level7_6_4/formal'); p.add_argument('--dry-run',action='store_true'); p.add_argument('--force',action='store_true'); args=p.parse_args()
    protocol={'variants':VARIANTS,'seeds':SEEDS,'stages':STAGES,'eval_length':EVAL_LENGTH,'windows':WINDOWS,'eval_samples_per_seed_window':args.eval_samples}
    if args.dry_run: print(json.dumps(protocol,indent=2)); return
    if args.eval_samples!=50: raise ValueError('Formal protocol locks eval_samples=50')
    if not torch.cuda.is_available(): raise RuntimeError('CUDA GPU required')
    device=torch.device('cuda'); dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16; target=params(build('ist-full',96)); width,_=matched_width(target); root=ROOT/args.output; root.mkdir(parents=True,exist_ok=True); runs=[]
    for variant in VARIANTS:
        for seed in SEEDS:
            runs.append(run_one(variant,seed,args,device,dtype,width,root)); atomic_save(root/'runs.partial.json',runs); torch.cuda.empty_cache()
    summary=[]
    for variant in VARIANTS:
        selected=[r for r in runs if r['variant']==variant]
        summary.append({'variant':variant,'trained_through_counts':{str(n):sum(r['trained_through']==n for r in selected) for n in (512,1024,2048,4096)},'mean_8192_accuracy_by_window':[sum(r['tests_8192'][i].get('correct',0) for r in selected)/sum(r['tests_8192'][i].get('samples',0) for r in selected) if sum(r['tests_8192'][i].get('samples',0) for r in selected) else None for i in range(len(WINDOWS))]})
    atomic_save(root/'result.json',{'protocol':protocol,'matched_width':width,'summary':summary,'runs':runs}); print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
