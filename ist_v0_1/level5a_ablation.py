import json, statistics, time
from pathlib import Path
import torch
import torch.nn.functional as F
from baseline_transformer import StandardTransformer
from long_context_test import set_seed
from marked_retrieval_level2 import make_batch
from model import InformationSpiralTransformer

SEEDS=[313,42]
STAGES=[(128,64,800),(256,128,250),(512,509,250)]
VARIANTS={
 "Transformer":{"kind":"baseline","diversity":0.0,"fusion":False},
 "IST-A":{"kind":"ist","diversity":0.0,"fusion":False},
 "IST-B":{"kind":"ist","diversity":0.1,"fusion":False},
 "IST-C":{"kind":"ist","diversity":0.1,"fusion":True},
}

def build(config):
    if config["kind"]=="baseline": return StandardTransformer(19,64,3,8,512,0.0,"rope")
    return InformationSpiralTransformer(19,64,3,512,"rope",config["fusion"])

@torch.no_grad()
def evaluate(model,length,needle_range,device,batches=10):
    model.eval(); correct=local_correct=total=0
    for _ in range(batches):
        x,y,pos=make_batch(32,length,needle_range,16,device); logits=model(x)[...,:16]
        rows=torch.arange(len(y),device=device); correct+=(logits[:,-1].argmax(-1)==y).sum().item()
        local_correct+=(logits[rows,pos].argmax(-1)==y).sum().item(); total+=len(y)
    return {"query_accuracy":correct/total,"local_accuracy":local_correct/total}

def run(name,config,seed,device):
    set_seed(seed); model=build(config).to(device); params=sum(p.numel() for p in model.parameters())
    set_seed(seed+10000); opt=torch.optim.AdamW(model.parameters(),lr=1e-3); curve=[]; global_step=0
    thresholds={"90":None,"95":None,"99":None}; started=time.perf_counter()
    if device.type=="cuda": torch.cuda.reset_peak_memory_stats(device)
    stages=[]
    for length,needle_range,steps in STAGES:
        for step in range(1,steps+1):
            global_step+=1; model.train(); x,y,pos=make_batch(32,length,needle_range,16,device)
            opt.zero_grad(set_to_none=True); logits=model(x)[...,:16]; rows=torch.arange(len(y),device=device)
            query=F.cross_entropy(logits[:,-1],y); local=F.cross_entropy(logits[rows,pos],y)
            loss=query+0.5*local
            if config["kind"]=="ist": loss=loss+config["diversity"]*model.memory_diversity_loss()
            loss.backward(); grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
            if global_step==1 or global_step%50==0:
                acc=(logits[:,-1].argmax(-1)==y).float().mean().item()
                curve.append({"step":global_step,"accuracy":acc,"query_loss":query.item(),"gradient_norm":float(grad)})
                for key,value in (("90",.90),("95",.95),("99",.99)):
                    if thresholds[key] is None and acc>=value: thresholds[key]=global_step
        metric=evaluate(model,length,needle_range,device); stages.append({"length":length,**metric})
    auc=sum((curve[i]["accuracy"]+curve[i-1]["accuracy"])/2*(curve[i]["step"]-curve[i-1]["step"]) for i in range(1,len(curve)))/global_step
    return {"model":name,"seed":seed,"parameters":params,"seconds":time.perf_counter()-started,
      "peak_memory_mb":torch.cuda.max_memory_allocated(device)/1048576 if device.type=="cuda" else None,
      "steps_to_accuracy":thresholds,"accuracy_auc":auc,"curve":curve,"stages":stages}

def main():
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); runs=[]
    for name,config in VARIANTS.items():
        for seed in SEEDS:
            print(f"running {name} seed={seed}",flush=True); result=run(name,config,seed,device); runs.append(result)
            print(result["stages"],flush=True); torch.cuda.empty_cache() if device.type=="cuda" else None
    summary=[]
    for name in VARIANTS:
        selected=[r for r in runs if r["model"]==name]; finals=[r["stages"][-1]["query_accuracy"] for r in selected]
        summary.append({"model":name,"parameters":selected[0]["parameters"],"mean_512_accuracy":statistics.mean(finals),
          "std_512_accuracy":statistics.stdev(finals),"mean_auc":statistics.mean(r["accuracy_auc"] for r in selected),
          "mean_seconds":statistics.mean(r["seconds"] for r in selected)})
    summary.sort(key=lambda x:(x["mean_512_accuracy"],x["mean_auc"]),reverse=True)
    out=Path("experiments/level5a"); out.mkdir(parents=True,exist_ok=True)
    (out/"results.json").write_text(json.dumps({"protocol":{"seeds":SEEDS,"stages":STAGES},"summary":summary,"runs":runs},indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2)); print(f"saved={out/'results.json'}")
if __name__=="__main__": main()
