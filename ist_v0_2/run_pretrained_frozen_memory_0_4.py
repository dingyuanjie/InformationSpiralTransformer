"""Frozen Memory 0.4: unique-stream 1K generalization across seeds."""
from __future__ import annotations
import argparse,json,math,random
from pathlib import Path
import torch
import torch.nn.functional as F
from config import HierarchicalMemoryConfig
from experiment_utils import ROOT,atomic_json,atomic_torch,run_metadata,parameter_count
from pretrained_memory_adapter import FrozenPretrainedIST,load_qwen
from run_pretrained_base_smoke import MODEL_ID,candidate_ids,make_tokens,alter
from run_pretrained_base_smoke_0_2 import diagnostics
from run_pretrained_base_smoke_0_3 import invariants
from run_pretrained_base_smoke_0_3_1 import fast_only_config

DISTANCE=1024;CHUNK=512;SEEDS=(313,42,2026)
def make_batch(tokenizer,seeds,split,device):
    rows=[make_tokens(tokenizer,s,split,DISTANCE) for s in seeds]
    return torch.stack([x[0] for x in rows]).to(device),torch.tensor([x[1] for x in rows],device=device)
@torch.no_grad()
def evaluate(backbone,adapter,tokenizer,labels,seeds,split,device,batch=4):
    output={key:[] for key in ("base","normal","zero_fast","reset_memory")};targets=[]
    for start in range(0,len(seeds),batch):
        ids,target=make_batch(tokenizer,seeds[start:start+batch],split,device);targets+=target.cpu().tolist();base=backbone(ids,use_cache=False).logits[:,-1,labels.to(device)].argmax(-1);output["base"]+=(base==target).int().cpu().tolist();parts=ids.split(CHUNK,dim=1);_,state=adapter(parts[0],None,detach_state=True)
        for condition in ("normal","zero_fast","reset_memory"):
            memory=alter(state,condition);intervention="zero_fast" if condition=="zero_fast" else "normal";logits,_=adapter(parts[1],memory,intervention=intervention,detach_state=True);prediction=logits[:,-1,labels.to(device)].argmax(-1);output[condition]+=(prediction==target).int().cpu().tolist()
    adapter.clear_intervention();return {"split":split,"samples":len(seeds),"correctness":output,"accuracy":{k:sum(v)/len(v) for k,v in output.items()},"targets":targets}
def train_seed(backbone,tokenizer,labels,seed,args,root,device,dtype):
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);adapter=FrozenPretrainedIST(backbone,fast_only_config(),identity_preserving=True,persistence_only=True).to(device=device,dtype=dtype);adapter.memory_scale.data=adapter.memory_scale.data.float();checks=invariants(backbone,adapter,tokenizer,device);memory_parameters=adapter.trainable_parameters();optimizer=torch.optim.AdamW([{"params":memory_parameters,"lr":8e-5},{"params":[adapter.memory_scale],"lr":2e-6}]);folder=root/f"seed{seed}";folder.mkdir(parents=True,exist_ok=True);resume=folder/"resume.pt";history=[];start=0;best={"score":-1e9,"step":0}
    if resume.exists() and not args.force:
        saved=torch.load(resume,map_location=device,weights_only=False);adapter.memory.load_state_dict(saved["memory"]);adapter.memory_scale.data.copy_(saved["memory_scale"].float());optimizer.load_state_dict(saved["optimizer"]);history=saved["history"];start=int(saved["step"]);best=saved["best"];print(f"resume seed={seed} step={start}",flush=True)
    validation_seeds=[110000000+seed*10000+i for i in range(args.validation_samples)]
    for step in range(start+1,args.steps+1):
        example_seeds=[120000000+seed*10000000+step*args.batch+i for i in range(args.batch)];ids,target=make_batch(tokenizer,example_seeds,"train",device);adapter.train();optimizer.zero_grad(set_to_none=True);state=None
        with torch.autocast(device_type="cuda",dtype=dtype):
            for part in ids.split(CHUNK,dim=1):logits,state=adapter(part,state)
            candidates=logits[:,-1,labels.to(device)];task=F.cross_entropy(candidates,target);teacher=adapter.last_base_logits[:,-1].float();student=logits[:,-1].float();distill=F.kl_div(F.log_softmax(student,dim=-1),F.softmax(teacher,dim=-1),reduction="batchmean");loss=task+.02*distill
        loss.backward();memory_grad=float(torch.nn.utils.clip_grad_norm_(memory_parameters,1));gate_grad=float(adapter.memory_scale.grad.abs()) if adapter.memory_scale.grad is not None else 0.0
        if adapter.memory_scale.grad is not None:adapter.memory_scale.grad.clamp_(-5,5)
        optimizer.step()
        if step==1 or step%25==0:
            row={"step":step,"loss":float(loss.detach()),"task_loss":float(task.detach()),"distill_loss":float(distill.detach()),"batch_accuracy":float((candidates.argmax(-1)==target).float().mean()),"raw_memory_grad":memory_grad,"raw_gate_grad":gate_grad,"diagnostics":diagnostics(adapter)}
            if step%args.validate_every==0:
                adapter.eval();validation=evaluate(backbone,adapter,tokenizer,labels,validation_seeds,"validation",device,args.eval_batch);score=validation["accuracy"]["normal"]-validation["accuracy"]["zero_fast"];row["validation"]=validation["accuracy"];row["validation_causal_gap"]=score
                if score>best["score"] or (score==best["score"] and validation["accuracy"]["normal"]>best.get("normal",-1)):
                    best={"score":score,"normal":validation["accuracy"]["normal"],"step":step};atomic_torch(folder/"best.pt",{"memory":adapter.memory.state_dict(),"memory_scale":adapter.memory_scale.detach(),"step":step,"validation":validation})
            history.append(row);print(f"seed={seed} "+json.dumps(row),flush=True);atomic_torch(resume,{"memory":adapter.memory.state_dict(),"memory_scale":adapter.memory_scale.detach(),"optimizer":optimizer.state_dict(),"history":history,"step":step,"best":best})
    selected=torch.load(folder/"best.pt",map_location=device,weights_only=False);adapter.memory.load_state_dict(selected["memory"]);adapter.memory_scale.data.copy_(selected["memory_scale"].float());heldout_seeds=[130000000+seed*10000+i for i in range(args.heldout_samples)];heldout=evaluate(backbone,adapter,tokenizer,labels,heldout_seeds,"held_out",device,args.eval_batch);return adapter,{"seed":seed,"invariants":checks,"best":best,"history":history,"heldout":heldout}
def wilson(k,n,z=1.95996398454):
    p=k/n;d=1+z*z/n;m=(p+z*z/(2*n))/d;h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d;return [m-h,m+h]
def paired_exact(treatment,control):
    improved=sum(a and not b for a,b in zip(treatment,control));harmed=sum(not a and b for a,b in zip(treatment,control));n=improved+harmed;tail=sum(math.comb(n,k) for k in range(min(improved,harmed)+1))/2**n if n else .5;return {"improved":improved,"harmed":harmed,"ties":len(control)-n,"mcnemar_exact_p":min(1.,2*tail),"difference":(sum(treatment)-sum(control))/len(control)}
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--model-id",default=MODEL_ID);p.add_argument("--seeds",nargs="+",type=int,default=list(SEEDS));p.add_argument("--steps",type=int,default=1000);p.add_argument("--batch",type=int,default=4);p.add_argument("--validation-samples",type=int,default=64);p.add_argument("--heldout-samples",type=int,default=128);p.add_argument("--eval-batch",type=int,default=4);p.add_argument("--validate-every",type=int,default=100);p.add_argument("--output",default="experiments/pretrained_base/frozen_memory_0_4/formal");p.add_argument("--dry-run",action="store_true");p.add_argument("--smoke-test",action="store_true");p.add_argument("--local-files-only",action="store_true");p.add_argument("--force",action="store_true");args=p.parse_args()
    if args.smoke_test:
        args.seeds=[2026];args.steps=2;args.batch=1;args.validation_samples=2;args.heldout_samples=2;args.eval_batch=1;args.validate_every=1
        if args.output.endswith("formal"):args.output=args.output[:-6]+"smoke"
    protocol={"model_id":args.model_id,"task":"unique-stream frozen Fast-Memory generalization","distance":DISTANCE,"seeds":args.seeds,"steps_per_seed":args.steps,"batch":args.batch,"unique_training_examples_per_seed":args.steps*args.batch,"validation_samples":args.validation_samples,"heldout_samples_per_seed":args.heldout_samples,"validate_every":args.validate_every,"checkpoint_selection":"max validation normal-minus-zero-fast, tie by normal","same_frozen_backbone":True,"fast_only":True,"persistence_only":True,"primary":"paired held-out normal vs zero-fast","chance":.25}
    if args.dry_run:print(json.dumps(protocol,indent=2));return 0
    if not torch.cuda.is_available():raise RuntimeError("CUDA required")
    device=torch.device("cuda");dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16;root=ROOT/args.output;root.mkdir(parents=True,exist_ok=True);tokenizer,backbone=load_qwen(args.model_id,dtype,device,args.local_files_only);labels=candidate_ids(tokenizer);runs=[]
    for seed in args.seeds:
        adapter,row=train_seed(backbone,tokenizer,labels,seed,args,root,device,dtype);runs.append(row);atomic_json(root/"runs.partial.json",runs);del adapter;torch.cuda.empty_cache()
    aggregate={condition:[v for run in runs for v in run["heldout"]["correctness"][condition]] for condition in ("base","normal","zero_fast","reset_memory")};summary={k:{"accuracy":sum(v)/len(v),"correct":sum(v),"samples":len(v),"wilson95":wilson(sum(v),len(v))} for k,v in aggregate.items()};paired_zero=paired_exact(aggregate["normal"],aggregate["zero_fast"]);paired_base=paired_exact(aggregate["normal"],aggregate["base"]);passed=summary["normal"]["wilson95"][0]>.25 and paired_zero["difference"]>0 and paired_zero["mcnemar_exact_p"]<.05;revision=getattr(backbone.config,"_commit_hash",None);protocol["resolved_revision"]=revision;result={"status":"complete","generalization_gate_passed":passed,"summary":summary,"paired_normal_vs_zero_fast":paired_zero,"paired_normal_vs_base":paired_base,"runs":runs,"protocol":protocol,"backbone_parameters":parameter_count(backbone)};atomic_json(root/"config.json",protocol);atomic_json(root/"run_metadata.json",run_metadata(device,args.seeds));atomic_json(root/"raw_results.json",result);atomic_json(root/"result.json",result);(root/"ANALYSIS.md").write_text("# Frozen Memory 0.4\n\nUnique-stream held-out generalization with paired causal evaluation.\n",encoding="utf-8");print(json.dumps({k:v for k,v in result.items() if k!="runs"},indent=2));return 0
if __name__=="__main__":raise SystemExit(main())
