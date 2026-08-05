import argparse, json, time
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from long_context_test import set_seed
from model import InformationSpiralTransformer

def make_batch(batch,size,device):
    target=torch.randint(16,(batch,),device=device)
    first=torch.randint(16,(batch,size),device=device);second=torch.randint(16,(batch,size),device=device)
    first[:,0]=17;first[:,1]=target;second[:,-2]=18;second[:,-1]=16
    return first,second,target

def memory_vector(memories):
    return torch.cat([memory.mean(dim=1) for memory in memories],dim=-1)

def forward_pair(model,probe,first,second):
    first_logits,memory1=model(first,return_memory=True,per_layer_memory=True)
    probe1=probe(memory_vector(memory1))
    second_logits,memory2=model(second,memory=memory1,return_memory=True,per_layer_memory=True)
    probe2=probe(memory_vector(memory2))
    return first_logits,second_logits,probe1,probe2,memory1,memory2

@torch.no_grad()
def evaluate(model,probe,args,device,dtype):
    model.eval();probe.eval();counts={"query":0,"local":0,"probe1":0,"probe2":0};total=0;similarities=[]
    for _ in range(args.eval_batches):
        first,second,target=make_batch(args.eval_batch_size,args.chunk_size,device)
        with torch.autocast(device_type="cuda",dtype=dtype):
            a,b,p1,p2,m1,m2=forward_pair(model,probe,first,second)
        counts["local"]+=(a[:,0,:16].argmax(-1)==target).sum().item()
        counts["query"]+=(b[:,-1,:16].argmax(-1)==target).sum().item()
        counts["probe1"]+=(p1.argmax(-1)==target).sum().item();counts["probe2"]+=(p2.argmax(-1)==target).sum().item()
        similarities.append(F.cosine_similarity(memory_vector(m1),memory_vector(m2),dim=-1).mean().item());total+=len(target)
    gates=[block.memory.last_diagnostics["update_gate"].float().mean().item() for block in model.blocks]
    return {**{key:value/total for key,value in counts.items()},"memory_cosine":sum(similarities)/len(similarities),"gate_means":gates,"samples":total}

def save(path,value):path.write_text(json.dumps(value,indent=2),encoding="utf-8")

def main():
    p=argparse.ArgumentParser(description="Level 6.1 minimal two-chunk memory diagnostic")
    p.add_argument("--seed",type=int,default=313);p.add_argument("--steps",type=int,default=2000);p.add_argument("--batch-size",type=int,default=16)
    p.add_argument("--eval-batch-size",type=int,default=16);p.add_argument("--eval-batches",type=int,default=10);p.add_argument("--eval-every",type=int,default=100)
    p.add_argument("--chunk-size",type=int,default=128);p.add_argument("--output",default="experiments/level6_1/formal");p.add_argument("--force",action="store_true");args=p.parse_args()
    if not torch.cuda.is_available():raise RuntimeError("CUDA GPU required")
    device=torch.device("cuda");dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    folder=Path(args.output)/f"persistent_seed{args.seed}";folder.mkdir(parents=True,exist_ok=True);final=folder/"result.json"
    if final.exists() and not args.force:print(f"completed: {final}");return
    set_seed(args.seed);model=InformationSpiralTransformer(19,64,3,args.chunk_size,"rope",True).to(device)
    probe=nn.Linear(64*3,16).to(device);set_seed(args.seed+10000)
    optimizer=torch.optim.AdamW(list(model.parameters())+list(probe.parameters()),lr=1e-3);history=[];passes=0;started=time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for step in range(1,args.steps+1):
        model.train();probe.train();first,second,target=make_batch(args.batch_size,args.chunk_size,device);optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda",dtype=dtype):
            a,b,p1,p2,_,_=forward_pair(model,probe,first,second)
            local=F.cross_entropy(a[:,0,:16],target);query=F.cross_entropy(b[:,-1,:16],target)
            loss=query+0.5*local+0.5*F.cross_entropy(p1,target)+0.5*F.cross_entropy(p2,target)+0.1*model.memory_diversity_loss()
        loss.backward();torch.nn.utils.clip_grad_norm_(list(model.parameters())+list(probe.parameters()),1.0);optimizer.step()
        if step==1 or step%args.eval_every==0:
            metric=evaluate(model,probe,args,device,dtype);history.append({"step":step,**metric});save(folder/"progress.json",history)
            print(f"step={step} query={metric['query']:.2%} p1={metric['probe1']:.2%} p2={metric['probe2']:.2%} local={metric['local']:.2%}",flush=True)
            passed=metric["query"]>=.95 and metric["probe1"]>=.95 and metric["probe2"]>=.95
            passes=passes+1 if passed else 0
            if passes>=2:break
    metric=evaluate(model,probe,args,device,dtype)
    result={"config":vars(args),"passed":passes>=2,"steps":step,"final":metric,"history":history,
            "seconds":time.perf_counter()-started,"peak_memory_mb":torch.cuda.max_memory_allocated()/1048576}
    save(final,result);torch.save({"model":model.state_dict(),"probe":probe.state_dict(),"result":result},folder/"checkpoint.pt")
    print(json.dumps(result["final"],indent=2));print("LEVEL6_1_PASS" if result["passed"] else "LEVEL6_1_FAIL")
if __name__=="__main__":main()
