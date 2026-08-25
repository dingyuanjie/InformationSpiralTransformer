"""Frozen Memory 0.4.1: nonzero fixed-gate warmup on a unique stream."""
from __future__ import annotations
import argparse,json,random
import torch
import torch.nn.functional as F
from experiment_utils import ROOT,atomic_json,atomic_torch,run_metadata,parameter_count
from pretrained_memory_adapter import FrozenPretrainedIST,load_qwen
from run_pretrained_base_smoke import MODEL_ID,candidate_ids
from run_pretrained_base_smoke_0_2 import diagnostics
from run_pretrained_base_smoke_0_3 import invariants
from run_pretrained_base_smoke_0_3_1 import fast_only_config
from run_pretrained_frozen_memory_0_4 import (SEEDS,CHUNK,make_batch,evaluate,wilson,paired_exact)

INITIAL_SCALE=-0.01
def train_seed(backbone,tokenizer,labels,seed,args,root,device,dtype):
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);adapter=FrozenPretrainedIST(backbone,fast_only_config(),identity_preserving=True,persistence_only=True).to(device=device,dtype=dtype);adapter.memory_scale.data=adapter.memory_scale.data.float();checks=invariants(backbone,adapter,tokenizer,device);adapter.memory_scale.data.fill_(INITIAL_SCALE);memory_parameters=adapter.trainable_parameters();optimizer=torch.optim.AdamW([{"params":memory_parameters,"lr":8e-5,"name":"fast_memory"},{"params":[adapter.memory_scale],"lr":0.0,"name":"gate"}]);folder=root/f"seed{seed}";folder.mkdir(parents=True,exist_ok=True);resume=folder/"resume.pt";history=[];start=0;best={"score":-1e9,"step":0}
    if resume.exists() and not args.force:
        saved=torch.load(resume,map_location=device,weights_only=False);adapter.memory.load_state_dict(saved["memory"]);adapter.memory_scale.data.copy_(saved["memory_scale"].float());optimizer.load_state_dict(saved["optimizer"]);history=saved["history"];start=int(saved["step"]);best=saved["best"];print(f"resume seed={seed} step={start}",flush=True)
    validation_seeds=[140000000+seed*10000+i for i in range(args.validation_samples)]
    for step in range(start+1,args.steps+1):
        gate_open=step>args.warmup_steps;optimizer.param_groups[1]["lr"]=args.gate_lr if gate_open else 0.0
        example_seeds=[150000000+seed*10000000+step*args.batch+i for i in range(args.batch)];ids,target=make_batch(tokenizer,example_seeds,"train",device);adapter.train();optimizer.zero_grad(set_to_none=True);state=None
        with torch.autocast(device_type="cuda",dtype=dtype):
            for part in ids.split(CHUNK,dim=1):logits,state=adapter(part,state)
            candidates=logits[:,-1,labels.to(device)];task=F.cross_entropy(candidates,target);teacher=adapter.last_base_logits[:,-1].float();student=logits[:,-1].float();distill=F.kl_div(F.log_softmax(student,dim=-1),F.softmax(teacher,dim=-1),reduction="batchmean");loss=task+args.distill_weight*distill
        loss.backward();memory_grad=float(torch.nn.utils.clip_grad_norm_(memory_parameters,1));gate_grad=float(adapter.memory_scale.grad.abs()) if adapter.memory_scale.grad is not None else 0.0
        if adapter.memory_scale.grad is not None:adapter.memory_scale.grad.clamp_(-5,5)
        optimizer.step()
        if not gate_open:adapter.memory_scale.data.fill_(INITIAL_SCALE)
        if step==1 or step%25==0:
            row={"step":step,"phase":"fixed_gate_warmup" if not gate_open else "joint","loss":float(loss.detach()),"task_loss":float(task.detach()),"distill_loss":float(distill.detach()),"batch_accuracy":float((candidates.argmax(-1)==target).float().mean()),"raw_memory_grad":memory_grad,"raw_gate_grad":gate_grad,"gate_lr":optimizer.param_groups[1]["lr"],"diagnostics":diagnostics(adapter)}
            if step%args.validate_every==0:
                adapter.eval();validation=evaluate(backbone,adapter,tokenizer,labels,validation_seeds,"validation",device,args.eval_batch);score=validation["accuracy"]["normal"]-validation["accuracy"]["zero_fast"];row["validation"]=validation["accuracy"];row["validation_causal_gap"]=score
                if score>best["score"] or (score==best["score"] and validation["accuracy"]["normal"]>best.get("normal",-1)):
                    best={"score":score,"normal":validation["accuracy"]["normal"],"step":step,"phase":row["phase"]};atomic_torch(folder/"best.pt",{"memory":adapter.memory.state_dict(),"memory_scale":adapter.memory_scale.detach(),"step":step,"validation":validation})
            history.append(row);print(f"seed={seed} "+json.dumps(row),flush=True);atomic_torch(resume,{"memory":adapter.memory.state_dict(),"memory_scale":adapter.memory_scale.detach(),"optimizer":optimizer.state_dict(),"history":history,"step":step,"best":best})
    selected=torch.load(folder/"best.pt",map_location=device,weights_only=False);adapter.memory.load_state_dict(selected["memory"]);adapter.memory_scale.data.copy_(selected["memory_scale"].float());heldout_seeds=[160000000+seed*10000+i for i in range(args.heldout_samples)];heldout=evaluate(backbone,adapter,tokenizer,labels,heldout_seeds,"held_out",device,args.eval_batch);return adapter,{"seed":seed,"invariants":checks,"best":best,"history":history,"heldout":heldout}
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--model-id",default=MODEL_ID);p.add_argument("--seeds",nargs="+",type=int,default=list(SEEDS));p.add_argument("--steps",type=int,default=1500);p.add_argument("--warmup-steps",type=int,default=500);p.add_argument("--batch",type=int,default=4);p.add_argument("--validation-samples",type=int,default=64);p.add_argument("--heldout-samples",type=int,default=128);p.add_argument("--eval-batch",type=int,default=4);p.add_argument("--validate-every",type=int,default=100);p.add_argument("--gate-lr",type=float,default=2e-6);p.add_argument("--distill-weight",type=float,default=.02);p.add_argument("--output",default="experiments/pretrained_base/frozen_memory_0_4_1/formal");p.add_argument("--dry-run",action="store_true");p.add_argument("--smoke-test",action="store_true");p.add_argument("--local-files-only",action="store_true");p.add_argument("--force",action="store_true");args=p.parse_args()
    if args.smoke_test:
        args.seeds=[2026];args.steps=2;args.warmup_steps=1;args.batch=1;args.validation_samples=2;args.heldout_samples=2;args.eval_batch=1;args.validate_every=1
        if args.output.endswith("formal"):args.output=args.output[:-6]+"smoke"
    protocol={"model_id":args.model_id,"task":"nonzero-gate unique-stream generalization","initial_scale":INITIAL_SCALE,"fresh_memory_per_seed":True,"no_fixed_set_weights":True,"seeds":args.seeds,"steps_per_seed":args.steps,"fixed_gate_warmup_steps":args.warmup_steps,"joint_steps":args.steps-args.warmup_steps,"batch":args.batch,"unique_training_examples_per_seed":args.steps*args.batch,"memory_lr":8e-5,"gate_lr_after_warmup":args.gate_lr,"distill_weight":args.distill_weight,"validation_samples":args.validation_samples,"heldout_samples_per_seed":args.heldout_samples,"checkpoint_selection":"max validation normal-minus-zero-fast, tie by normal","primary":"paired held-out normal vs zero-fast","chance":.25}
    if args.dry_run:print(json.dumps(protocol,indent=2));return 0
    if not torch.cuda.is_available():raise RuntimeError("CUDA required")
    device=torch.device("cuda");dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16;root=ROOT/args.output;root.mkdir(parents=True,exist_ok=True);tokenizer,backbone=load_qwen(args.model_id,dtype,device,args.local_files_only);labels=candidate_ids(tokenizer);runs=[]
    for seed in args.seeds:
        adapter,row=train_seed(backbone,tokenizer,labels,seed,args,root,device,dtype);runs.append(row);atomic_json(root/"runs.partial.json",runs);del adapter;torch.cuda.empty_cache()
    aggregate={condition:[v for run in runs for v in run["heldout"]["correctness"][condition]] for condition in ("base","normal","zero_fast","reset_memory")};summary={k:{"accuracy":sum(v)/len(v),"correct":sum(v),"samples":len(v),"wilson95":wilson(sum(v),len(v))} for k,v in aggregate.items()};paired_zero=paired_exact(aggregate["normal"],aggregate["zero_fast"]);paired_base=paired_exact(aggregate["normal"],aggregate["base"]);passed=summary["normal"]["wilson95"][0]>.25 and paired_zero["difference"]>0 and paired_zero["mcnemar_exact_p"]<.05;revision=getattr(backbone.config,"_commit_hash",None);protocol["resolved_revision"]=revision;result={"status":"complete","generalization_gate_passed":passed,"summary":summary,"paired_normal_vs_zero_fast":paired_zero,"paired_normal_vs_base":paired_base,"runs":runs,"protocol":protocol,"backbone_parameters":parameter_count(backbone)};atomic_json(root/"config.json",protocol);atomic_json(root/"run_metadata.json",run_metadata(device,args.seeds));atomic_json(root/"raw_results.json",result);atomic_json(root/"result.json",result);(root/"ANALYSIS.md").write_text("# Frozen Memory 0.4.1\n\nNonzero fixed-gate warmup, then low-LR gate training on a unique stream.\n",encoding="utf-8");print(json.dumps({k:v for k,v in result.items() if k!="runs"},indent=2));return 0
if __name__=="__main__":raise SystemExit(main())
