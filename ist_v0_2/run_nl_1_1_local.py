"""NL-1.1: short-range learnability calibration before long-memory claims."""
from __future__ import annotations
import argparse,json,math
import torch
import torch.nn.functional as F
from experiment_utils import ROOT,atomic_json,atomic_torch,parameter_count,run_metadata
from natural_language_data import generate_nl1,audit_example
from run_nl_1_local import ARCHITECTURES,build,forward_stream,set_seed

SEEDS=(313,42,2026,7,1234);DISTANCES=(512,1024,2048);CHUNK_SIZE=512
STAGES=((512,300,8),(1024,250,4),(2048,200,2));EVAL_SAMPLES=64;CHANCE=.25

def examples(ids,split,distance,device):
    rows=[generate_nl1(i,split,distance,CHUNK_SIZE,option_count=4) for i in ids]
    chunks=[torch.tensor([r.chunks[j] for r in rows],device=device) for j in range(distance//CHUNK_SIZE)]
    return chunks,torch.tensor([r.target for r in rows],device=device),rows
def train(model,arch,seed,folder,device,dtype,force,stages):
    optimizer=torch.optim.AdamW(model.parameters(),lr=7e-4);history=[]
    for stage,(distance,steps,batch) in enumerate(stages,1):
        final=folder/f"stage{stage}.pt";resume=folder/f"stage{stage}_resume.pt";start=0
        if final.exists() and not force:
            state=torch.load(final,map_location=device,weights_only=False);model.load_state_dict(state["model"]);optimizer.load_state_dict(state["optimizer"]);history=state["history"];continue
        if resume.exists() and not force:
            state=torch.load(resume,map_location=device,weights_only=False);model.load_state_dict(state["model"]);optimizer.load_state_dict(state["optimizer"]);history=state["history"];start=int(state["step"]);print(f"resume arch={arch} seed={seed} stage={stage} step={start}",flush=True)
        for step in range(start+1,steps+1):
            ids=[850000000+seed*100000+stage*10000+step*16+i for i in range(batch)]
            chunks,target,_=examples(ids,"train",distance,device);model.train();optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda",dtype=dtype):
                logits,_=forward_stream(model,arch,chunks);loss=F.cross_entropy(logits[:,-1],target)+.05*model.memory_diversity_loss()
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step()
            if step==1 or step%20==0:
                acc=float((logits[:,-1].argmax(-1)==target).float().mean());row={"stage":stage,"distance":distance,"step":step,"loss":float(loss.detach()),"accuracy":acc};history.append(row);print(f"arch={arch} seed={seed} distance={distance} step={step} loss={row['loss']:.4f} accuracy={acc:.2%}",flush=True);atomic_torch(resume,{"model":model.state_dict(),"optimizer":optimizer.state_dict(),"history":history,"step":step})
        atomic_torch(final,{"model":model.state_dict(),"optimizer":optimizer.state_dict(),"history":history,"step":steps})
    return history
@torch.no_grad()
def evaluate(model,arch,seed,device,distances,samples):
    model.eval();out=[]
    for split in ("validation","held_out","ood"):
        for distance in distances:
            correct=[];predictions=[];targets=[];audits=[]
            for offset in range(0,samples,16):
                ids=[860000000+seed*100000+distance+offset+i for i in range(min(16,samples-offset))];chunks,target,rows=examples(ids,split,distance,device);logits,_=forward_stream(model,arch,chunks,detach=True);prediction=logits[:,-1].argmax(-1);correct+=(prediction==target).int().cpu().tolist();predictions+=prediction.cpu().tolist();targets+=target.cpu().tolist();audits += [audit_example(r) for r in rows]
            row={"split":split,"distance":distance,"correctness":correct,"predictions":predictions,"targets":targets,"audits":audits,"accuracy":sum(correct)/len(correct)};out.append(row);print(f"arch={arch} seed={seed} split={split} distance={distance} accuracy={row['accuracy']:.2%}",flush=True)
    return out
def wilson(k,n,z=1.95996398454):
    p=k/n;d=1+z*z/n;m=(p+z*z/(2*n))/d;h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d;return [m-h,m+h]
def summarize(runs):
    out=[]
    for arch in ARCHITECTURES:
      for split in ("validation","held_out","ood"):
       for distance in DISTANCES:
        values=[v for run in runs if run["architecture"]==arch for row in run["rows"] if row["split"]==split and row["distance"]==distance for v in row["correctness"]];ci=wilson(sum(values),len(values));out.append({"architecture":arch,"split":split,"distance":distance,"correct":sum(values),"samples":len(values),"accuracy":sum(values)/len(values),"wilson95":ci,"above_chance":ci[0]>CHANCE})
    return out
def plot(summary,root):
    import matplotlib.pyplot as plt
    fig,axis=plt.subplots(figsize=(7,5))
    for arch in ARCHITECTURES:
        rows=[r for r in summary if r["architecture"]==arch and r["split"]=="held_out"];axis.plot([r["distance"] for r in rows],[r["accuracy"] for r in rows],marker="o",label=arch)
    axis.axhline(CHANCE,color="gray",linestyle="--",label="chance");axis.set_xscale("log",base=2);axis.set_ylim(0,1);axis.set_xlabel("Token distance");axis.set_ylabel("Accuracy");axis.grid(alpha=.25);axis.legend();fig.tight_layout();fig.savefig(root/"short_range_learnability.png",dpi=180);plt.close(fig)
def main():
    global STAGES,DISTANCES,EVAL_SAMPLES
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--seeds",nargs="+",type=int,default=list(SEEDS));p.add_argument("--output",default="experiments/natural_language/nl_1_1_learnability/formal");p.add_argument("--dry-run",action="store_true");p.add_argument("--smoke-test",action="store_true");p.add_argument("--force",action="store_true");args=p.parse_args()
    if args.smoke_test:
        args.seeds=[2026];STAGES=((512,3,2),);DISTANCES=(512,1024);EVAL_SAMPLES=8
        if args.output.endswith("formal"):args.output=args.output[:-6]+"smoke"
    protocol={"task":"NL-1.1 short-range learnability","architectures":ARCHITECTURES,"seeds":args.seeds,"distances":DISTANCES,"stages":STAGES,"splits":["validation","held_out","ood"],"options":4,"chance":CHANCE,"samples_per_seed":EVAL_SAMPLES,"gate":"held-out Wilson lower bound above chance at 512 and 1K"}
    if args.dry_run:print(json.dumps(protocol,indent=2));return 0
    if not torch.cuda.is_available():raise RuntimeError("NL-1.1 requires CUDA")
    device=torch.device("cuda");dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16;root=ROOT/args.output;root.mkdir(parents=True,exist_ok=True);atomic_json(root/"config.json",protocol);atomic_json(root/"run_metadata.json",run_metadata(device,args.seeds));runs=[];training=[]
    for arch in ARCHITECTURES:
      for seed in args.seeds:
        set_seed(seed);folder=root/arch/f"seed{seed}";folder.mkdir(parents=True,exist_ok=True);model=build(arch,device);history=train(model,arch,seed,folder,device,dtype,args.force,STAGES);training.append({"architecture":arch,"seed":seed,"parameters":parameter_count(model),"history":history});path=folder/"evaluation.json";rows=json.loads(path.read_text(encoding="utf-8")) if path.exists() and not args.force else evaluate(model,arch,seed,device,DISTANCES,EVAL_SAMPLES);atomic_json(path,rows);runs.append({"architecture":arch,"seed":seed,"rows":rows});atomic_json(root/"runs.partial.json",runs);del model;torch.cuda.empty_cache()
    summary=summarize(runs);required=tuple(d for d in (512,1024) if d in DISTANCES);gate=any(all(any(r["architecture"]==arch and r["split"]=="held_out" and r["distance"]==distance and r["above_chance"] for r in summary) for distance in required) for arch in ("ist_v0_1","ist_v0_2"));result={"status":"complete","learnability_gate_passed":gate,"protocol":protocol,"summary":summary,"training":training,"runs":runs};atomic_json(root/"raw_results.json",result);atomic_json(root/"result.json",result);plot(summary,root);(root/"ANALYSIS.md").write_text("# NL-1.1 Analysis\n\nRun complete. The gate is statistical short-range learnability, not long-memory superiority.\n",encoding="utf-8");print(json.dumps({"status":"complete","learnability_gate_passed":gate,"summary":summary},indent=2));return 0
if __name__=="__main__":raise SystemExit(main())
