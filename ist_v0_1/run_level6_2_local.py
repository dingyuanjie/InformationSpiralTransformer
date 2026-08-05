import argparse, json, time
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from long_context_test import set_seed
from model import InformationSpiralTransformer

def make_chunks(batch,count,size,device):
    target=torch.randint(16,(batch,),device=device);data=torch.randint(16,(batch,count,size),device=device)
    pos=torch.randint(0,size-2,(batch,),device=device);rows=torch.arange(batch,device=device)
    data[rows,0,pos]=17;data[rows,0,pos+1]=target;data[:,-1,-2]=18;data[:,-1,-1]=16
    return data,target,pos

def vector(memories):return torch.cat([memory.mean(dim=1) for memory in memories],dim=-1)

def forward_chunks(model,probe,chunks):
    memory=None;first_logits=None;probe_logits=[];memory_vectors=[]
    for index in range(chunks.size(1)):
        logits,memory=model(chunks[:,index],memory=memory,return_memory=True,per_layer_memory=True)
        if index==0:first_logits=logits
        state=vector(memory);memory_vectors.append(state);probe_logits.append(probe(state))
    return logits,first_logits,probe_logits,memory_vectors

@torch.no_grad()
def evaluate(model,probe,args,count,device,dtype,batches=None):
    model.eval();probe.eval();batches=batches or args.eval_batches;query=local=first_probe=final_probe=total=0
    per_chunk=[0]*count;similarities=[]
    for _ in range(batches):
        chunks,target,pos=make_chunks(args.eval_batch_size,count,args.chunk_size,device)
        with torch.autocast(device_type="cuda",dtype=dtype):last,first,probes,states=forward_chunks(model,probe,chunks)
        rows=torch.arange(len(target),device=device);query+=(last[:,-1,:16].argmax(-1)==target).sum().item()
        local+=(first[rows,pos,:16].argmax(-1)==target).sum().item()
        for i,item in enumerate(probes):per_chunk[i]+=(item.argmax(-1)==target).sum().item()
        first_probe+=(probes[0].argmax(-1)==target).sum().item();final_probe+=(probes[-1].argmax(-1)==target).sum().item()
        similarities.append(F.cosine_similarity(states[0],states[-1],dim=-1).mean().item());total+=len(target)
    accuracies=[value/total for value in per_chunk]
    gates=[block.memory.last_diagnostics["update_gate"].float().mean().item() for block in model.blocks]
    return {"chunks":count,"total_tokens":count*args.chunk_size,"query":query/total,"local":local/total,
      "probe_first":first_probe/total,"probe_final":final_probe/total,"probe_min":min(accuracies),
      "probe_by_chunk":accuracies,"memory_first_final_cosine":sum(similarities)/len(similarities),"gate_means":gates,"samples":total}

def save(path,value):path.write_text(json.dumps(value,indent=2),encoding="utf-8")

def main():
    p=argparse.ArgumentParser(description="Level 6.2 random-marker multi-chunk curriculum")
    p.add_argument("--seed",type=int,default=313);p.add_argument("--chunk-size",type=int,default=128);p.add_argument("--batch-size",type=int,default=8)
    p.add_argument("--stage1-steps",type=int,default=800);p.add_argument("--later-steps",type=int,default=500);p.add_argument("--eval-every",type=int,default=100)
    p.add_argument("--eval-batch-size",type=int,default=8);p.add_argument("--eval-batches",type=int,default=10);p.add_argument("--probe-weight",type=float,default=0.5)
    p.add_argument("--init-checkpoint",default="experiments/level6_1/formal/persistent_seed313/checkpoint.pt")
    p.add_argument("--resume-checkpoint",default=None)
    p.add_argument("--output",default="experiments/level6_2/formal");p.add_argument("--force",action="store_true");args=p.parse_args()
    if not torch.cuda.is_available():raise RuntimeError("CUDA GPU required")
    device=torch.device("cuda");dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    folder=Path(args.output)/f"persistent_seed{args.seed}";folder.mkdir(parents=True,exist_ok=True);final=folder/"result.json"
    if final.exists() and not args.force:print(f"completed: {final}");return
    set_seed(args.seed);model=InformationSpiralTransformer(19,64,3,args.chunk_size,"rope",True).to(device);probe=nn.Linear(192,16).to(device)
    if args.init_checkpoint and Path(args.init_checkpoint).exists():
        checkpoint=torch.load(args.init_checkpoint,map_location=device,weights_only=False);model.load_state_dict(checkpoint["model"]);probe.load_state_dict(checkpoint["probe"])
        print(f"initialized={args.init_checkpoint}",flush=True)
    set_seed(args.seed+20000);optimizer=torch.optim.AdamW(list(model.parameters())+list(probe.parameters()),lr=1e-3)
    history=[];stages=[];start_stage=0
    if args.resume_checkpoint:
        checkpoint=torch.load(args.resume_checkpoint,map_location=device,weights_only=False)
        model.load_state_dict(checkpoint["model"]);probe.load_state_dict(checkpoint["probe"]);optimizer.load_state_dict(checkpoint["optimizer"])
        history=checkpoint["history"];stages=checkpoint["stages"];start_stage=len(stages)
        print(f"resumed={args.resume_checkpoint} after_stage={start_stage}",flush=True)
    started=time.perf_counter();torch.cuda.reset_peak_memory_stats();specs=[(2,args.stage1_steps),(4,args.later_steps),(8,args.later_steps),(16,args.later_steps)]
    lr_scales={2:1.0,4:1.0,8:0.25,16:0.1}
    for stage,(count,steps) in enumerate(specs[start_stage:],start_stage+1):
        for group in optimizer.param_groups:group["lr"]=1e-3*lr_scales[count]
        batch=max(1,min(args.batch_size,32//count));passes=0
        for step in range(1,steps+1):
            model.train();probe.train();chunks,target,pos=make_chunks(batch,count,args.chunk_size,device);optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda",dtype=dtype):
                last,first,probes,_=forward_chunks(model,probe,chunks);rows=torch.arange(batch,device=device)
                query=F.cross_entropy(last[:,-1,:16],target);local=F.cross_entropy(first[rows,pos,:16],target)
                probe_loss=torch.stack([F.cross_entropy(item,target) for item in probes]).mean()
                loss=query+0.5*local+args.probe_weight*probe_loss+0.1*model.memory_diversity_loss()
            loss.backward();torch.nn.utils.clip_grad_norm_(list(model.parameters())+list(probe.parameters()),1.0);optimizer.step()
            if step==1 or step%args.eval_every==0:
                metric=evaluate(model,probe,args,count,device,dtype);history.append({"stage":stage,"step":step,**metric});save(folder/"progress.json",history)
                print(f"chunks={count} step={step} query={metric['query']:.2%} probe_final={metric['probe_final']:.2%} probe_min={metric['probe_min']:.2%}",flush=True)
                passed=metric["query"]>=.95 and metric["probe_first"]>=.95 and metric["probe_final"]>=.95 and metric["probe_min"]>=.90
                passes=passes+1 if passed else 0
                if passes>=2:break
        stages.append({"chunks":count,"steps":step,"passed":passes>=2,"validation":metric})
        torch.save({"model":model.state_dict(),"probe":probe.state_dict(),"optimizer":optimizer.state_dict(),"history":history,"stages":stages},folder/f"stage{stage}.pt")
        if passes<2:break
    result={"config":vars(args),"passed":len(stages)==4 and all(s["passed"] for s in stages),"stages":stages,"history":history,
      "seconds":time.perf_counter()-started,"peak_memory_mb":torch.cuda.max_memory_allocated()/1048576};save(final,result)
    print("LEVEL6_2_PASS" if result["passed"] else "LEVEL6_2_FAIL")
if __name__=="__main__":main()
