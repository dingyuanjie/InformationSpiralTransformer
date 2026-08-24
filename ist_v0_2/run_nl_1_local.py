"""NL-1 formal bridge: natural-language single-fact retrieval with causal controls."""
from __future__ import annotations
import argparse, json, math
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import HierarchicalMemoryConfig
from experiment_utils import ROOT, atomic_json, atomic_torch, parameter_count, run_metadata
from model import build_model
from natural_language_data import VOCAB_SIZE, generate_nl1, audit_example

SEEDS=(313,42,2026,7,1234); DISTANCES=(2048,4096,8192,16384,32768); CHUNK_SIZE=512
STAGES=((2048,120,4),(4096,100,2),(8192,80,1)); EVAL_SAMPLES=32
ARCHITECTURES=("transformer","ist_v0_1","ist_v0_2")
V02_CONDITIONS=("normal","zero_memory","reset_memory","roll_memory","zero_fast","zero_slow","zero_episodic")

class ChunkTransformer(nn.Module):
    memory_arch="transformer"
    def __init__(self,hidden=64,layers=3):
        super().__init__();self.embedding=nn.Embedding(VOCAB_SIZE,hidden)
        block=nn.TransformerEncoderLayer(hidden,8,hidden*4,batch_first=True,norm_first=True)
        self.blocks=nn.TransformerEncoder(block,layers);self.output=nn.Linear(hidden,VOCAB_SIZE)
        positions=torch.arange(CHUNK_SIZE).float()[:,None];frequencies=torch.exp(torch.arange(0,hidden,2).float()*(-math.log(10000.0)/hidden))
        encoding=torch.zeros(CHUNK_SIZE,hidden);encoding[:,0::2]=torch.sin(positions*frequencies);encoding[:,1::2]=torch.cos(positions*frequencies)
        self.register_buffer("position_encoding",encoding,persistent=False)
    def forward(self,tokens,memory=None,return_memory=False,**kwargs):
        hidden=self.embedding(tokens)+self.position_encoding[:tokens.size(1)][None].to(self.embedding.weight.dtype)
        logits=self.output(self.blocks(hidden));return (logits,None) if return_memory else logits
    def memory_diversity_loss(self):return self.embedding.weight.new_zeros(())

def set_seed(seed):torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
def build(arch,device):
    if arch=="transformer":return ChunkTransformer().to(device)
    if arch=="ist_v0_1":return build_model("v0_1",vocab_size=VOCAB_SIZE,hidden_size=64,layers=3,max_sequence_length=512,position_encoding="rope",use_memory_fusion=True).to(device)
    return build_model("hierarchical_v0_2",vocab_size=VOCAB_SIZE,hidden_size=64,layers=3,max_sequence_length=512,position_encoding="rope",hierarchical_config=HierarchicalMemoryConfig()).to(device)
def batch_examples(seeds,split,distance,device):
    examples=[generate_nl1(s,split,distance,512) for s in seeds]
    chunks=[torch.tensor([e.chunks[i] for e in examples],device=device) for i in range(distance//512)]
    return chunks,torch.tensor([e.target for e in examples],device=device),examples
def forward_stream(model,arch,chunks,memory=None,detach=False):
    for tokens in chunks:
        if arch=="ist_v0_1":logits,memory=model(tokens,memory=memory,return_memory=True,per_layer_memory=True,detach_memory=detach)
        elif arch=="ist_v0_2":logits,memory=model(tokens,memory=memory,return_memory=True,detach_memory=detach)
        else:logits,memory=model(tokens,return_memory=True)
    return logits,memory
def clone_memory(memory):
    if memory is None:return None
    if isinstance(memory[0],dict):return [{k:v.clone() if torch.is_tensor(v) else v for k,v in layer.items()} for layer in memory]
    return [v.clone() for v in memory]
def alter_memory(memory,condition):
    if memory is None or condition=="normal":return clone_memory(memory)
    if condition=="reset_memory":return None
    result=clone_memory(memory)
    if isinstance(result[0],dict):
        for layer in result:
            keys=("fast","slow","episodic_keys","episodic_values")
            if condition=="zero_memory":
                for key in keys:layer[key].zero_()
            elif condition=="roll_memory":
                for key in keys:layer[key]=torch.roll(layer[key],1,1)
    elif condition=="zero_memory":result=[torch.zeros_like(v) for v in result]
    elif condition=="roll_memory":result=[torch.roll(v,1,1) for v in result]
    return result
def train(model,arch,seed,folder,device,dtype,force,stages):
    optimizer=torch.optim.AdamW(model.parameters(),lr=5e-4);history=[]
    for stage,(distance,steps,batch) in enumerate(stages,1):
        final=folder/f"stage{stage}.pt";resume=folder/f"stage{stage}_resume.pt";start=0
        if final.exists() and not force:
            state=torch.load(final,map_location=device,weights_only=False);model.load_state_dict(state["model"]);optimizer.load_state_dict(state["optimizer"]);history=state["history"];continue
        if resume.exists() and not force:
            state=torch.load(resume,map_location=device,weights_only=False);model.load_state_dict(state["model"]);optimizer.load_state_dict(state["optimizer"]);history=state["history"];start=int(state["step"])
            print(f"resume arch={arch} seed={seed} stage={stage} step={start}",flush=True)
        for step in range(start+1,steps+1):
            ids=[830000000+seed*100000+stage*10000+step*16+i for i in range(batch)]
            chunks,target,_=batch_examples(ids,"train",distance,device);model.train();optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda",dtype=dtype):
                logits,_=forward_stream(model,arch,chunks);loss=F.cross_entropy(logits[:,-1],target)+.05*model.memory_diversity_loss()
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step()
            if step==1 or step%20==0:
                acc=float((logits[:,-1].argmax(-1)==target).float().mean());row={"stage":stage,"distance":distance,"step":step,"loss":float(loss.detach()),"accuracy":acc};history.append(row)
                print(f"arch={arch} seed={seed} distance={distance} step={step} loss={row['loss']:.4f} accuracy={acc:.2%}",flush=True)
                atomic_torch(resume,{"model":model.state_dict(),"optimizer":optimizer.state_dict(),"history":history,"step":step})
        atomic_torch(final,{"model":model.state_dict(),"optimizer":optimizer.state_dict(),"history":history})
    return history
@torch.no_grad()
def evaluate(model,arch,seed,device,distances,samples):
    model.eval();rows=[]
    conditions=("normal",) if arch=="transformer" else (("normal","zero_memory","reset_memory","roll_memory") if arch=="ist_v0_1" else V02_CONDITIONS)
    for split in ("validation","held_out","ood"):
        for distance in distances:
            for offset in range(0,samples,8):
                count=min(8,samples-offset);ids=[840000000+seed*100000+distance+offset+i for i in range(count)]
                chunks,target,examples=batch_examples(ids,split,distance,device);_,memory=forward_stream(model,arch,chunks[:-1],detach=True)
                for condition in conditions:
                    if arch=="ist_v0_2":model.set_memory_intervention(condition if condition.startswith("zero_") and condition!="zero_memory" else "normal")
                    logits,_=forward_stream(model,arch,chunks[-1:],alter_memory(memory,condition),detach=True);prediction=logits[:,-1].argmax(-1)
                    rows.append({"split":split,"distance":distance,"condition":condition,"correctness":(prediction==target).int().cpu().tolist(),"predictions":prediction.cpu().tolist(),"targets":target.cpu().tolist(),"example_seeds":ids,"audits":[audit_example(e) for e in examples]})
                if arch=="ist_v0_2":model.clear_memory_interventions()
            values=[v for r in rows if r["split"]==split and r["distance"]==distance and r["condition"]=="normal" for v in r["correctness"]]
            print(f"arch={arch} seed={seed} split={split} distance={distance} normal={sum(values)/len(values):.2%}",flush=True)
    return rows
def summarize(runs):
    keys=sorted({(run["architecture"],r["split"],r["distance"],r["condition"]) for run in runs for r in run["rows"]});out=[]
    for arch,split,distance,condition in keys:
        values=[v for run in runs if run["architecture"]==arch for r in run["rows"] if (r["split"],r["distance"],r["condition"])==(split,distance,condition) for v in r["correctness"]]
        out.append({"architecture":arch,"split":split,"distance":distance,"condition":condition,"correct":sum(values),"samples":len(values),"accuracy":sum(values)/len(values)})
    return out
def write_curve(summary,root):
    import matplotlib.pyplot as plt
    curves=(("transformer","normal","Transformer"),("ist_v0_1","normal","IST v0.1"),("ist_v0_2","normal","IST v0.2"),("ist_v0_2","zero_memory","IST v0.2 zero-memory"))
    fig,axis=plt.subplots(figsize=(8,5))
    for arch,condition,label in curves:
        rows=sorted((r for r in summary if r["split"]=="held_out" and r["architecture"]==arch and r["condition"]==condition),key=lambda r:r["distance"])
        if rows:axis.plot([r["distance"] for r in rows],[r["accuracy"] for r in rows],marker="o",label=label)
    axis.set_xscale("log",base=2);axis.set_ylim(0,1);axis.set_xlabel("Token distance");axis.set_ylabel("Accuracy");axis.grid(alpha=.25);axis.legend();fig.tight_layout();fig.savefig(root/"accuracy_vs_token_distance.png",dpi=180);plt.close(fig)
def main():
    global STAGES,DISTANCES,EVAL_SAMPLES
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--seeds",nargs="+",type=int,default=list(SEEDS));p.add_argument("--output",default="experiments/natural_language/nl_1_single_fact/formal");p.add_argument("--dry-run",action="store_true");p.add_argument("--smoke-test",action="store_true");p.add_argument("--force",action="store_true");args=p.parse_args()
    if args.smoke_test:
        args.seeds=[2026];STAGES=((2048,2,2),);DISTANCES=(2048,4096);EVAL_SAMPLES=8
        if args.output.endswith("formal"):args.output=args.output[:-6]+"smoke"
    protocol={"task":"NL-1 single-fact retrieval","architectures":ARCHITECTURES,"seeds":args.seeds,"distances":DISTANCES,"splits":["train","validation","held_out","ood"],"stages":STAGES,"evaluation_samples_per_seed":EVAL_SAMPLES,"v0_2_conditions":V02_CONDITIONS,"tokenizer":"lossless UTF-8 bytes","vocab_size":VOCAB_SIZE,"chunk_size":512}
    if args.dry_run:print(json.dumps(protocol,indent=2));return 0
    if not torch.cuda.is_available():raise RuntimeError("NL-1 requires CUDA")
    device=torch.device("cuda");dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16;root=ROOT/args.output;root.mkdir(parents=True,exist_ok=True);atomic_json(root/"config.json",protocol);atomic_json(root/"run_metadata.json",run_metadata(device,args.seeds));runs=[];training=[]
    for arch in ARCHITECTURES:
        for seed in args.seeds:
            set_seed(seed);folder=root/arch/f"seed{seed}";folder.mkdir(parents=True,exist_ok=True);model=build(arch,device);history=train(model,arch,seed,folder,device,dtype,args.force,STAGES);training.append({"architecture":arch,"seed":seed,"parameters":parameter_count(model),"history":history});output=folder/"evaluation.json"
            rows=json.loads(output.read_text(encoding="utf-8")) if output.exists() and not args.force else evaluate(model,arch,seed,device,DISTANCES,EVAL_SAMPLES)
            atomic_json(output,rows);runs.append({"architecture":arch,"seed":seed,"rows":rows});atomic_json(root/"runs.partial.json",runs);del model;torch.cuda.empty_cache()
    summary=summarize(runs);result={"status":"complete","protocol":protocol,"summary":summary,"training":training,"runs":runs};atomic_json(root/"raw_results.json",result);atomic_json(root/"result.json",result);write_curve(summary,root);(root/"ANALYSIS.md").write_text("# NL-1 Analysis\n\nGate on held-out/OOD curves and paired Memory interventions.\n",encoding="utf-8");print(json.dumps({"status":"complete","summary":summary},indent=2));return 0
if __name__=="__main__":raise SystemExit(main())
