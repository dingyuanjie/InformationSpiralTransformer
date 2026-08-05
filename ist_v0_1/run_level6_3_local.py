import argparse,json,time
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from long_context_test import set_seed
from model import InformationSpiralTransformer
from run_level6_2_local import make_chunks,forward_chunks,evaluate

def save(path,value):path.write_text(json.dumps(value,indent=2),encoding="utf-8")

def main():
    p=argparse.ArgumentParser(description="Level 6.3 probe supervision withdrawal")
    p.add_argument("--seed",type=int,default=313);p.add_argument("--chunk-size",type=int,default=128);p.add_argument("--chunks",type=int,default=16)
    p.add_argument("--batch-size",type=int,default=2);p.add_argument("--eval-batch-size",type=int,default=8);p.add_argument("--eval-batches",type=int,default=10)
    p.add_argument("--eval-every",type=int,default=50);p.add_argument("--lr",type=float,default=1e-4)
    p.add_argument("--checkpoint",default="experiments/level6_2/formal/persistent_seed313/stage4.pt")
    p.add_argument("--resume-checkpoint",default=None)
    p.add_argument("--output",default="experiments/level6_3/formal");p.add_argument("--force",action="store_true");args=p.parse_args()
    if not torch.cuda.is_available():raise RuntimeError("CUDA GPU required")
    device=torch.device("cuda");dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    folder=Path(args.output)/f"persistent_seed{args.seed}";folder.mkdir(parents=True,exist_ok=True);final=folder/"result.json"
    if final.exists() and not args.force:print(f"completed: {final}");return
    set_seed(args.seed);model=InformationSpiralTransformer(19,64,3,args.chunk_size,"rope",True).to(device);probe=nn.Linear(192,16).to(device)
    checkpoint=torch.load(args.checkpoint,map_location=device,weights_only=False);model.load_state_dict(checkpoint["model"]);probe.load_state_dict(checkpoint["probe"])
    set_seed(args.seed+30000);optimizer=torch.optim.AdamW(list(model.parameters())+list(probe.parameters()),lr=args.lr)
    eval_args=argparse.Namespace(eval_batch_size=args.eval_batch_size,eval_batches=args.eval_batches,chunk_size=args.chunk_size)
    schedule=[(0.2,300),(0.1,300),(0.0,500)];history=[];stages=[];start_stage=0
    if args.resume_checkpoint:
        resumed=torch.load(args.resume_checkpoint,map_location=device,weights_only=False)
        model.load_state_dict(resumed["model"]);probe.load_state_dict(resumed["probe"]);optimizer.load_state_dict(resumed["optimizer"])
        stages=resumed["stages"];start_stage=len(stages)
        progress=folder/"progress.json"
        if progress.exists():history=json.loads(progress.read_text(encoding="utf-8"))
        print(f"resumed={args.resume_checkpoint} after_stage={start_stage}",flush=True)
    started=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    baseline=evaluate(model,probe,eval_args,args.chunks,device,dtype);print(f"baseline query={baseline['query']:.2%} probe_min={baseline['probe_min']:.2%}",flush=True)
    for stage,(weight,steps) in enumerate(schedule[start_stage:],start_stage+1):
        for parameter in probe.parameters():parameter.requires_grad_(weight>0)
        for step in range(1,steps+1):
            model.train();probe.train(weight>0);chunks,target,pos=make_chunks(args.batch_size,args.chunks,args.chunk_size,device);optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda",dtype=dtype):
                last,first,probes,_=forward_chunks(model,probe,chunks);rows=torch.arange(args.batch_size,device=device)
                query=F.cross_entropy(last[:,-1,:16],target);local=F.cross_entropy(first[rows,pos,:16],target)
                probe_loss=torch.stack([F.cross_entropy(item,target) for item in probes]).mean()
                loss=query+0.5*local+weight*probe_loss+0.1*model.memory_diversity_loss()
            loss.backward();torch.nn.utils.clip_grad_norm_(list(model.parameters())+list(probe.parameters()),1.0);optimizer.step()
            if step==1 or step%args.eval_every==0:
                metric=evaluate(model,probe,eval_args,args.chunks,device,dtype);history.append({"stage":stage,"probe_weight":weight,"step":step,**metric});save(folder/"progress.json",history)
                print(f"weight={weight} step={step} query={metric['query']:.2%} probe_min={metric['probe_min']:.2%} probe_final={metric['probe_final']:.2%}",flush=True)
        stages.append({"probe_weight":weight,"steps":steps,"validation":metric})
        torch.save({"model":model.state_dict(),"probe":probe.state_dict(),"optimizer":optimizer.state_dict(),"stages":stages},folder/f"stage{stage}.pt")
    final_metric=evaluate(model,probe,eval_args,args.chunks,device,dtype)
    passed=final_metric["query"]>=.95 and final_metric["probe_min"]>=.90
    result={"config":vars(args),"baseline":baseline,"stages":stages,"final":final_metric,"passed":passed,"history":history,
      "seconds":time.perf_counter()-started,"peak_memory_mb":torch.cuda.max_memory_allocated()/1048576};save(final,result)
    print(json.dumps(final_metric,indent=2));print("LEVEL6_3_PASS" if passed else "LEVEL6_3_FAIL")
if __name__=="__main__":main()
