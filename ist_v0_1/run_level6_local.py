import argparse, json, statistics, time
from pathlib import Path
import torch
import torch.nn.functional as F
from baseline_transformer import StandardTransformer
from long_context_test import set_seed
from model import InformationSpiralTransformer

VARIANTS=("transformer","ist-persistent","ist-reset")

def make_chunks(batch,chunks,size,vocab,device):
    mask,needle,query=vocab,vocab+1,vocab+2
    target=torch.randint(vocab,(batch,),device=device)
    data=torch.randint(vocab,(batch,chunks,size),device=device)
    pos=torch.randint(0,size-2,(batch,),device=device); rows=torch.arange(batch,device=device)
    data[rows,0,pos]=needle; data[rows,0,pos+1]=target
    data[:,-1,-2]=query; data[:,-1,-1]=mask
    return data,target,pos

def build(name,size):
    if name=="transformer": return StandardTransformer(19,64,3,8,size,0.0,"rope")
    return InformationSpiralTransformer(19,64,3,size,"rope",True)

def forward_chunks(model,name,chunks,detach_between=False):
    memory=None; first_logits=None
    for index in range(chunks.size(1)):
        if name=="transformer": logits=model(chunks[:,index])
        else:
            incoming=None if name=="ist-reset" else memory
            logits,memory=model(chunks[:,index],memory=incoming,return_memory=True,
                                detach_memory=detach_between)
        if index==0: first_logits=logits
    return logits,first_logits,memory

@torch.no_grad()
def evaluate(model,name,args,chunk_count,device,dtype,batches=10):
    model.eval(); correct=total=0; started=time.perf_counter()
    for _ in range(batches):
        chunks,target,_=make_chunks(args.eval_batch_size,chunk_count,args.chunk_size,16,device)
        with torch.autocast(device_type="cuda",dtype=dtype):
            logits,_,_=forward_chunks(model,name,chunks,detach_between=True)
        correct+=(logits[:,-1,:16].argmax(-1)==target).sum().item(); total+=len(target)
    torch.cuda.synchronize(); seconds=time.perf_counter()-started
    return {"chunks":chunk_count,"total_tokens":chunk_count*args.chunk_size,
            "accuracy":correct/total,"samples":total,"seconds":seconds,
            "tokens_per_second":total*chunk_count*args.chunk_size/seconds}

def save(path,obj): path.write_text(json.dumps(obj,indent=2),encoding="utf-8")

def train_one(name,seed,args,device,dtype,root):
    folder=root/f"{name}_seed{seed}"; folder.mkdir(parents=True,exist_ok=True); final=folder/"result.json"
    if final.exists() and not args.force: return json.loads(final.read_text(encoding="utf-8"))
    set_seed(seed); model=build(name,args.chunk_size).to(device); set_seed(seed+10000)
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3); history=[]; started=time.perf_counter(); start_stage=0
    checkpoints=sorted(folder.glob("stage*.pt"))
    if checkpoints and not args.force:
        checkpoint=torch.load(checkpoints[-1],map_location=device,weights_only=False)
        model.load_state_dict(checkpoint["model"]);optimizer.load_state_dict(checkpoint["optimizer"])
        torch.set_rng_state(checkpoint["torch_rng_state"]);torch.cuda.set_rng_state(checkpoint["cuda_rng_state"])
        history=checkpoint["history"];start_stage=int(checkpoints[-1].stem.replace("stage",""))
        print(f"resume {folder} after stage {start_stage}",flush=True)
    if device.type=="cuda": torch.cuda.reset_peak_memory_stats()
    stages=[(2,args.stage1_steps),(4,args.later_steps),(8,args.later_steps),(16,args.later_steps)]
    for stage,(chunk_count,steps) in enumerate(stages[start_stage:],start_stage+1):
        stage_batch=max(1,min(args.batch_size,16//chunk_count))
        for step in range(1,steps+1):
            model.train(); chunks,target,pos=make_chunks(stage_batch,chunk_count,args.chunk_size,16,device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda",dtype=dtype):
                logits,first,_=forward_chunks(model,name,chunks,detach_between=False)
                rows=torch.arange(stage_batch,device=device)
                query=F.cross_entropy(logits[:,-1,:16],target)
                local=F.cross_entropy(first[rows,pos,:16],target)
                loss=query+0.5*local
                if name.startswith("ist-"): loss=loss+0.1*model.memory_diversity_loss()
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); optimizer.step()
            if step%args.eval_every==0:
                metric=evaluate(model,name,args,chunk_count,device,dtype,args.eval_batches)
                history.append({"stage":stage,"step":step,**metric}); save(folder/"progress.json",history)
                print(f"{name} seed={seed} chunks={chunk_count} step={step} acc={metric['accuracy']:.2%}",flush=True)
        torch.save({"model":model.state_dict(),"optimizer":optimizer.state_dict(),"history":history,
                    "torch_rng_state":torch.get_rng_state(),"cuda_rng_state":torch.cuda.get_rng_state()},folder/f"stage{stage}.pt")
    tests=[evaluate(model,name,args,n,device,dtype,args.long_eval_batches) for n in (2,4,8,16)]
    result={"variant":name,"seed":seed,"parameters":sum(p.numel() for p in model.parameters()),
            "seconds":time.perf_counter()-started,"peak_memory_mb":torch.cuda.max_memory_allocated()/1048576,
            "history":history,"tests":tests}; save(final,result); return result

def main():
    p=argparse.ArgumentParser(description="Level 6 cross-chunk persistent memory")
    p.add_argument("--variants",nargs="+",choices=VARIANTS,default=list(VARIANTS));p.add_argument("--seeds",nargs="+",type=int,default=[313,42])
    p.add_argument("--chunk-size",type=int,default=512);p.add_argument("--batch-size",type=int,default=8);p.add_argument("--eval-batch-size",type=int,default=2)
    p.add_argument("--stage1-steps",type=int,default=800);p.add_argument("--later-steps",type=int,default=400);p.add_argument("--eval-every",type=int,default=100)
    p.add_argument("--eval-batches",type=int,default=5);p.add_argument("--long-eval-batches",type=int,default=10)
    p.add_argument("--output",default="experiments/level6/formal");p.add_argument("--force",action="store_true");args=p.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError("CUDA GPU required")
    device=torch.device("cuda");dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    root=Path(args.output);root.mkdir(parents=True,exist_ok=True);results=[]
    print(f"GPU={torch.cuda.get_device_name()} chunk={args.chunk_size}",flush=True)
    for name in args.variants:
        for seed in args.seeds:
            results.append(train_one(name,seed,args,device,dtype,root));save(root/"runs.partial.json",results);torch.cuda.empty_cache()
    summary=[]
    for name in args.variants:
        selected=[r for r in results if r["variant"]==name]
        summary.append({"variant":name,"parameters":selected[0]["parameters"],
          "mean_accuracy":{str(n):statistics.mean(r["tests"][i]["accuracy"] for r in selected) for i,n in enumerate((2,4,8,16))},
          "mean_seconds":statistics.mean(r["seconds"] for r in selected),"mean_memory_mb":statistics.mean(r["peak_memory_mb"] for r in selected)})
    save(root/"summary.json",{"protocol":vars(args),"summary":summary,"runs":results});print(json.dumps(summary,indent=2))
if __name__=="__main__":main()
