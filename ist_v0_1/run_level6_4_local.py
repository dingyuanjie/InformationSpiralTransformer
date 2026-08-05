import argparse,json,statistics,time
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from long_context_test import set_seed
from model import InformationSpiralTransformer
from run_level6_2_local import make_chunks,forward_chunks,evaluate

SEEDS=[313,42,2026,7,1234]
def save(path,value):path.write_text(json.dumps(value,indent=2),encoding="utf-8")

def build(device,size):return InformationSpiralTransformer(19,64,3,size,"rope",True).to(device),nn.Linear(192,16).to(device)

def train_step(model,probe,optimizer,batch,count,size,device,dtype,probe_weight):
    chunks,target,pos=make_chunks(batch,count,size,device);optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda",dtype=dtype):
        last,first,probes,_=forward_chunks(model,probe,chunks);rows=torch.arange(batch,device=device)
        query=F.cross_entropy(last[:,-1,:16],target);local=F.cross_entropy(first[rows,pos,:16],target)
        probe_loss=torch.stack([F.cross_entropy(item,target) for item in probes]).mean()
        loss=query+0.5*local+probe_weight*probe_loss+0.1*model.memory_diversity_loss()
    loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);optimizer.step()

def maintenance(seed,args,device,dtype,root):
    folder=root/f"maintenance_seed{seed}";folder.mkdir(parents=True,exist_ok=True);final=folder/"result.json"
    if final.exists() and not args.force:return json.loads(final.read_text(encoding="utf-8"))
    model,probe=build(device,args.chunk_size);checkpoint=torch.load(args.checkpoint,map_location=device,weights_only=False)
    model.load_state_dict(checkpoint["model"]);probe.load_state_dict(checkpoint["probe"])
    for parameter in probe.parameters():parameter.requires_grad_(False)
    set_seed(seed+40000);optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4);history=[];started=time.perf_counter()
    for step in range(1,args.maintenance_steps+1):
        model.train();train_step(model,probe,optimizer,2,16,args.chunk_size,device,dtype,0.0)
        if step%args.eval_every==0:
            metric=evaluate(model,probe,args,16,device,dtype);history.append({"step":step,**metric});save(folder/"progress.json",history)
            print(f"maintenance seed={seed} step={step} query={metric['query']:.2%} probe_min={metric['probe_min']:.2%}",flush=True)
    final_metric=evaluate(model,probe,args,16,device,dtype,args.final_eval_batches)
    result={"mode":"maintenance","seed":seed,"passed":final_metric["query"]>=.95 and final_metric["probe_min"]>=.90,
            "final":final_metric,"history":history,"seconds":time.perf_counter()-started};save(final,result);return result

def scratch(seed,args,device,dtype,root):
    folder=root/f"scratch_zero_seed{seed}";folder.mkdir(parents=True,exist_ok=True);final=folder/"result.json"
    if final.exists() and not args.force:return json.loads(final.read_text(encoding="utf-8"))
    set_seed(seed);model,probe=build(device,args.chunk_size)
    for parameter in probe.parameters():parameter.requires_grad_(False)
    set_seed(seed+50000);optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3);history=[];stages=[];started=time.perf_counter()
    specs=[(2,args.scratch_stage1_steps,1e-3,8),(4,args.scratch_later_steps,1e-3,4),(8,args.scratch_later_steps,2.5e-4,4),(16,args.scratch_later_steps,1e-4,2)]
    for stage,(count,steps,lr,batch) in enumerate(specs,1):
        for group in optimizer.param_groups:group["lr"]=lr
        passes=0
        for step in range(1,steps+1):
            model.train();train_step(model,probe,optimizer,batch,count,args.chunk_size,device,dtype,0.0)
            if step%args.eval_every==0:
                metric=evaluate(model,probe,args,count,device,dtype);history.append({"stage":stage,"step":step,**metric});save(folder/"progress.json",history)
                print(f"scratch seed={seed} chunks={count} step={step} query={metric['query']:.2%}",flush=True)
                passes=passes+1 if metric["query"]>=.95 else 0
                if passes>=2:break
        stages.append({"chunks":count,"steps":step,"passed":passes>=2,"validation":metric})
        torch.save({"model":model.state_dict(),"optimizer":optimizer.state_dict(),"stages":stages,"history":history},folder/f"stage{stage}.pt")
        if passes<2:break
    result={"mode":"scratch-zero","seed":seed,"passed":len(stages)==4 and all(s["passed"] for s in stages),
            "stages":stages,"history":history,"seconds":time.perf_counter()-started};save(final,result);return result

def main():
    p=argparse.ArgumentParser(description="Level 6.4 maintenance and spontaneous-memory tests")
    p.add_argument("--modes",nargs="+",choices=["maintenance","scratch-zero"],default=["maintenance","scratch-zero"])
    p.add_argument("--seeds",nargs="+",type=int,default=SEEDS);p.add_argument("--chunk-size",type=int,default=128)
    p.add_argument("--maintenance-steps",type=int,default=500);p.add_argument("--scratch-stage1-steps",type=int,default=3000);p.add_argument("--scratch-later-steps",type=int,default=1000)
    p.add_argument("--eval-every",type=int,default=100);p.add_argument("--eval-batch-size",type=int,default=8);p.add_argument("--eval-batches",type=int,default=10);p.add_argument("--final-eval-batches",type=int,default=50)
    p.add_argument("--checkpoint",default="experiments/level6_2/formal/persistent_seed313/stage4.pt");p.add_argument("--output",default="experiments/level6_4/formal");p.add_argument("--force",action="store_true");args=p.parse_args()
    if not torch.cuda.is_available():raise RuntimeError("CUDA GPU required")
    device=torch.device("cuda");dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16;root=Path(args.output);root.mkdir(parents=True,exist_ok=True);results=[]
    for mode in args.modes:
        for seed in args.seeds:
            result=maintenance(seed,args,device,dtype,root) if mode=="maintenance" else scratch(seed,args,device,dtype,root)
            results.append(result);save(root/"runs.partial.json",results);torch.cuda.empty_cache()
    summary=[]
    for mode in args.modes:
        selected=[r for r in results if r["mode"]==mode];summary.append({"mode":mode,"successes":sum(r["passed"] for r in selected),"runs":len(selected),"success_rate":statistics.mean(r["passed"] for r in selected),"mean_seconds":statistics.mean(r["seconds"] for r in selected)})
    save(root/"summary.json",{"protocol":vars(args),"summary":summary,"runs":results});print(json.dumps(summary,indent=2))
if __name__=="__main__":main()
