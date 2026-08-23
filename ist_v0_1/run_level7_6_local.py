import argparse, json, statistics, time
from pathlib import Path
import torch
import torch.nn.functional as F
from baseline_transformer import StandardTransformer
from long_context_test import set_seed
from marked_retrieval_level2 import make_batch
from model import InformationSpiralTransformer

SEEDS=[313,42,2026,7,1234]
STAGES=[(128,64,2000),(256,128,800),(512,509,800)]
LONG_LENGTHS=(1024,2048,4096,8192)

def params(model): return sum(p.numel() for p in model.parameters())

def matched_width(target):
    candidates=[]
    for width in range(64,129,8):
        count=params(StandardTransformer(19,width,3,8,8192,0.0,"rope"))
        candidates.append((abs(count-target),width,count))
    return min(candidates)[1:]

def build(name,matched):
    if name=="transformer-64": return StandardTransformer(19,64,3,8,8192,0.0,"rope")
    if name=="transformer-matched": return StandardTransformer(19,matched,3,8,8192,0.0,"rope")
    model=InformationSpiralTransformer(19,64,3,8192,"rope",True)
    if name=="ist-stable": model.blocks[2].memory.slot_queries.requires_grad_(False)
    return model

@torch.no_grad()
def evaluate(model,length,needle_range,batches,batch_size,device,dtype):
    model.eval(); correct=local=total=0; loss_sum=seconds=0.0
    if device.type=="cuda": torch.cuda.synchronize()
    started=time.perf_counter()
    for _ in range(batches):
        x,y,pos=make_batch(batch_size,length,needle_range,16,device)
        with torch.autocast(device_type=device.type,dtype=dtype,enabled=device.type=="cuda"):
            logits=model(x)[...,:16]; q=F.cross_entropy(logits[:,-1],y)
        rows=torch.arange(len(y),device=device); correct+=(logits[:,-1].argmax(-1)==y).sum().item()
        local+=(logits[rows,pos].argmax(-1)==y).sum().item(); total+=len(y); loss_sum+=q.item()
    if device.type=="cuda": torch.cuda.synchronize()
    seconds=time.perf_counter()-started
    return {"query_accuracy":correct/total,"local_accuracy":local/total,"query_loss":loss_sum/batches,
            "samples":total,"seconds":seconds,"tokens_per_second":total*length/seconds}

def save(path,value): path.write_text(json.dumps(value,indent=2),encoding="utf-8")

def run(name,seed,args,device,dtype,matched,root):
    folder=root/f"{name}_seed{seed}"; folder.mkdir(parents=True,exist_ok=True); final=folder/"result.json"
    if final.exists() and not args.force: return json.loads(final.read_text(encoding="utf-8"))
    set_seed(seed); model=build(name,matched).to(device); optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3)
    set_seed(seed+10000); history=[]; stages=[]; global_step=0; started=time.perf_counter(); start_stage=0
    checkpoints=sorted(folder.glob("stage*.pt"))
    if checkpoints and not args.force:
        checkpoint=torch.load(checkpoints[-1],map_location=device,weights_only=False)
        model.load_state_dict(checkpoint["model"]); optimizer.load_state_dict(checkpoint["optimizer"])
        torch.set_rng_state(checkpoint["torch_rng_state"])
        if device.type=="cuda": torch.cuda.set_rng_state(checkpoint["cuda_rng_state"])
        stages=checkpoint["stages"]; start_stage=len(stages); global_step=sum(s["steps"] for s in stages)
        progress=folder/"progress.json"
        if progress.exists(): history=json.loads(progress.read_text(encoding="utf-8"))
        print(f"resume {folder} after stage {start_stage}",flush=True)
    if device.type=="cuda": torch.cuda.reset_peak_memory_stats()
    for index,(length,needle_range,steps) in enumerate(STAGES[start_stage:],start_stage+1):
        for step in range(1,steps+1):
            global_step+=1; model.train(); x,y,pos=make_batch(args.batch_size,length,needle_range,16,device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,dtype=dtype,enabled=device.type=="cuda"):
                logits=model(x)[...,:16]; rows=torch.arange(len(y),device=device)
                q=F.cross_entropy(logits[:,-1],y); local=F.cross_entropy(logits[rows,pos],y)
                loss=q+0.5*local+(0.1*model.memory_diversity_loss() if name.startswith("ist-") else 0)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); optimizer.step()
            if step%args.eval_every==0:
                metric=evaluate(model,length,needle_range,args.eval_batches,args.eval_batch_size,device,dtype)
                history.append({"global_step":global_step,"stage":index,"stage_step":step,"length":length,**metric})
                print(f"{name} seed={seed} L={length} step={step} acc={metric['query_accuracy']:.2%}",flush=True)
        stages.append({"length":length,"steps":steps,"validation":history[-1]})
        torch.save({"model":model.state_dict(),"optimizer":optimizer.state_dict(),"stages":stages,
                    "torch_rng_state":torch.get_rng_state(),"cuda_rng_state":torch.cuda.get_rng_state()},folder/f"stage{index}.pt")
        save(folder/"progress.json",history)
    long_tests=[]
    for n in LONG_LENGTHS:
        try: long_tests.append({"length":n,**evaluate(model,n,n-3,args.long_eval_batches,1,device,dtype),"status":"ok"})
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache(); long_tests.append({"length":n,"status":"oom"})
    auc=sum((history[i]["query_accuracy"]+history[i-1]["query_accuracy"])/2*(history[i]["global_step"]-history[i-1]["global_step"]) for i in range(1,len(history)))/global_step
    result={"variant":name,"seed":seed,"parameters":params(model),"trainable_parameters":sum(p.numel() for p in model.parameters() if p.requires_grad),"l3_slot_frozen":bool(name=="ist-stable" and not model.blocks[2].memory.slot_queries.requires_grad),"accuracy_auc":auc,"seconds":time.perf_counter()-started,
            "peak_memory_mb":torch.cuda.max_memory_allocated()/1048576,"stages":stages,"long_tests":long_tests,"history":history}
    save(final,result); return result

def main():
    p=argparse.ArgumentParser(description="Level 7.6 stable-IST external long-context validation")
    p.add_argument("--variants",nargs="+",default=["transformer-matched","ist-full","ist-stable"],choices=["transformer-64","transformer-matched","ist-full","ist-stable"])
    p.add_argument("--seeds",nargs="+",type=int,default=SEEDS); p.add_argument("--batch-size",type=int,default=16)
    p.add_argument("--eval-batch-size",type=int,default=16); p.add_argument("--eval-batches",type=int,default=10); p.add_argument("--eval-every",type=int,default=100)
    p.add_argument("--long-eval-batch-size",type=int,default=4); p.add_argument("--long-eval-batches",type=int,default=10)
    p.add_argument("--output",default="experiments/level7_6/formal"); p.add_argument("--force",action="store_true"); p.add_argument("--dry-run",action="store_true"); args=p.parse_args()
    if args.dry_run: print(json.dumps(vars(args),indent=2)); return
    if not torch.cuda.is_available(): raise RuntimeError("CUDA GPU required")
    device=torch.device("cuda"); dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    target=params(InformationSpiralTransformer(19,64,3,8192,"rope",True)); matched,matched_params=matched_width(target)
    root=Path(args.output); root.mkdir(parents=True,exist_ok=True); results=[]
    print(f"GPU={torch.cuda.get_device_name()} IST_params={target} matched_width={matched} matched_params={matched_params}")
    for name in args.variants:
        for seed in args.seeds:
            results.append(run(name,seed,args,device,dtype,matched,root)); save(root/"runs.partial.json",results); torch.cuda.empty_cache()
    summary=[]
    for name in args.variants:
        selected=[r for r in results if r["variant"]==name]
        summary.append({"variant":name,"parameters":selected[0]["parameters"],"mean_auc":statistics.mean(r["accuracy_auc"] for r in selected),
          "mean_time":statistics.mean(r["seconds"] for r in selected),"mean_memory_mb":statistics.mean(r["peak_memory_mb"] for r in selected),
          "mean_long_accuracy":{str(n):statistics.mean([r["long_tests"][index]["query_accuracy"] for r in selected if r["long_tests"][index]["status"]=="ok"]) if any(r["long_tests"][index]["status"]=="ok" for r in selected) else None
                                for index,n in enumerate(LONG_LENGTHS)}})
    save(root/"summary.json",{"protocol":vars(args),"matched_width":matched,"summary":summary,"runs":results}); print(json.dumps(summary,indent=2))
if __name__=="__main__": main()
