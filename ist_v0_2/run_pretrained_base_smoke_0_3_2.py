"""Base Smoke 0.3.2: continue 0.3.1 with separately conditioned optimization."""
from __future__ import annotations
import argparse,json,random
from pathlib import Path
import torch
import torch.nn.functional as F
from config import HierarchicalMemoryConfig
from experiment_utils import ROOT,atomic_json,atomic_torch,run_metadata,parameter_count
from pretrained_memory_adapter import FrozenPretrainedIST,load_qwen
from run_pretrained_base_smoke import MODEL_ID,candidate_ids
from run_pretrained_base_smoke_0_2 import fixed_data,batch_of,grad_norm,score,diagnostics
from run_pretrained_base_smoke_0_3 import invariants
from run_pretrained_base_smoke_0_3_1 import fast_only_config

SOURCE=ROOT/"experiments/pretrained_base/base_smoke_0_3_1/formal/memory_checkpoint.pt"
def continue_train(adapter,labels,data,additional_steps,batch,device,dtype,root,source):
    memory_parameters=adapter.trainable_parameters();optimizer=torch.optim.AdamW([{"params":memory_parameters,"lr":8e-5},{"params":[adapter.memory_scale],"lr":2e-6}]);history=[];start=0;resume=root/"continuation_resume.pt"
    if resume.exists():
        saved=torch.load(resume,map_location=device,weights_only=False);adapter.memory.load_state_dict(saved["memory"]);adapter.memory_scale.data.copy_(saved["memory_scale"].float());optimizer.load_state_dict(saved["optimizer"]);history=saved["history"];start=int(saved["continuation_step"]);print(f"resume continuation_step={start}",flush=True)
    else:
        saved=torch.load(source,map_location=device,weights_only=False);adapter.memory.load_state_dict(saved["memory"]);adapter.memory_scale.data.copy_(saved["memory_scale"].float());print("loaded Base Smoke 0.3.1 step-1000 checkpoint",flush=True)
    stable=0
    for step in range(start+1,additional_steps+1):
        rng=random.Random(100000000+step);indices=[rng.randrange(len(data)) for _ in range(batch)];ids,target=batch_of(data,indices);adapter.train();optimizer.zero_grad(set_to_none=True);state=None
        with torch.autocast(device_type="cuda",dtype=dtype):
            for part in ids.split(512,dim=1):logits,state=adapter(part,state)
            candidates=logits[:,-1,labels.to(device)];task=F.cross_entropy(candidates,target);teacher=adapter.last_base_logits[:,-1].float();student=logits[:,-1].float();distill=F.kl_div(F.log_softmax(student,dim=-1),F.softmax(teacher,dim=-1),reduction="batchmean");loss=task+.02*distill
        loss.backward();raw_gate=float(adapter.memory_scale.grad.abs()) if adapter.memory_scale.grad is not None else 0.0;memory_grad=float(torch.nn.utils.clip_grad_norm_(memory_parameters,1.0));
        if adapter.memory_scale.grad is not None:adapter.memory_scale.grad.clamp_(-5,5)
        optimizer.step()
        if step==1 or step%25==0:
            row={"continuation_step":step,"total_step":1000+step,"loss":float(loss.detach()),"task_loss":float(task.detach()),"distill_loss":float(distill.detach()),"batch_accuracy":float((candidates.argmax(-1)==target).float().mean()),"raw_memory_grad":memory_grad,"raw_gate_grad":raw_gate,"clamped_gate_grad":min(raw_gate,5.0),"diagnostics":diagnostics(adapter)}
            if step%50==0:
                adapter.eval();row["fixed_normal"]=score(adapter,labels,data,"normal",batch);row["fixed_zero_memory"]=score(adapter,labels,data,"zero_memory",batch);row["fixed_reset_memory"]=score(adapter,labels,data,"reset_memory",batch);row["fixed_zero_fast"]=score(adapter,labels,data,"zero_fast",batch);row["causal_gap"]=row["fixed_normal"]-max(row["fixed_zero_memory"],row["fixed_reset_memory"]);row["zero_fast_equals_reset"]=row["fixed_zero_fast"]==row["fixed_reset_memory"];stable=stable+1 if row["fixed_normal"]>=.95 and row["causal_gap"]>=.5 and row["zero_fast_equals_reset"] else 0;row["stable_checks"]=stable
            history.append(row);print(json.dumps(row),flush=True);atomic_torch(resume,{"memory":adapter.memory.state_dict(),"memory_scale":adapter.memory_scale.detach(),"optimizer":optimizer.state_dict(),"history":history,"continuation_step":step})
            if stable>=2:print("FAST_PERSISTENCE_GATE_PASS",flush=True);break
    return history,optimizer,history[-1]["continuation_step"]
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--model-id",default=MODEL_ID);p.add_argument("--additional-steps",type=int,default=1000);p.add_argument("--batch",type=int,default=4);p.add_argument("--source",type=Path,default=SOURCE);p.add_argument("--output",default="experiments/pretrained_base/base_smoke_0_3_2/formal");p.add_argument("--dry-run",action="store_true");p.add_argument("--local-files-only",action="store_true");args=p.parse_args();protocol={"model_id":args.model_id,"source_checkpoint":str(args.source),"source_total_step":1000,"additional_steps":args.additional_steps,"batch":args.batch,"memory_lr":8e-5,"gate_lr":2e-6,"memory_clip_norm":1.0,"gate_gradient_clamp":5.0,"fast_only":True,"persistence_only":True,"gate":{"fixed_normal":.95,"causal_gap":.5,"zero_fast_equals_reset":True,"consecutive_checks":2},"not_a_generalization_result":True}
    if args.dry_run:print(json.dumps(protocol,indent=2));return 0
    if not args.source.exists():raise FileNotFoundError(f"source checkpoint missing: {args.source}")
    if not torch.cuda.is_available():raise RuntimeError("CUDA required")
    device=torch.device("cuda");dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16;root=ROOT/args.output;root.mkdir(parents=True,exist_ok=True);tokenizer,backbone=load_qwen(args.model_id,dtype,device,args.local_files_only);labels=candidate_ids(tokenizer);adapter=FrozenPretrainedIST(backbone,fast_only_config(),identity_preserving=True,persistence_only=True).to(device=device,dtype=dtype);adapter.memory_scale.data=adapter.memory_scale.data.float();checks=invariants(backbone,adapter,tokenizer,device);data=fixed_data(tokenizer,device)
    history,optimizer,completed=continue_train(adapter,labels,data,args.additional_steps,args.batch,device,dtype,root,args.source);conditions=("normal","zero_memory","reset_memory","roll_memory","zero_fast");fixed={c:score(adapter,labels,data,c,args.batch) for c in conditions};held=fixed_data(tokenizer,device,101000000,"held_out");heldout={c:score(adapter,labels,held,c,args.batch) for c in conditions};gap=fixed["normal"]-max(fixed["zero_memory"],fixed["reset_memory"]);zero_consistent=fixed["zero_fast"]==fixed["reset_memory"];passed=fixed["normal"]>=.95 and gap>=.5 and zero_consistent;revision=getattr(backbone.config,"_commit_hash",None);protocol["resolved_revision"]=revision;result={"status":"complete","fast_persistence_gate_passed":passed,"invariants":checks,"source_step":1000,"completed_additional_steps":completed,"total_steps":1000+completed,"fixed_train":fixed,"heldout_diagnostic":heldout,"causal_gap":gap,"zero_fast_equals_reset":zero_consistent,"final_diagnostics":diagnostics(adapter),"history":history,"protocol":protocol,"backbone_parameters":parameter_count(backbone),"trainable_parameters":sum(p.numel() for p in adapter.trainable_parameters())+1};atomic_json(root/"config.json",protocol);atomic_json(root/"run_metadata.json",run_metadata(device,102000001));atomic_torch(root/"memory_checkpoint.pt",{"memory":adapter.memory.state_dict(),"memory_scale":adapter.memory_scale.detach(),"optimizer":optimizer.state_dict(),"history":history,"revision":revision,"total_steps":1000+completed});atomic_json(root/"raw_results.json",result);atomic_json(root/"result.json",result);(root/"ANALYSIS.md").write_text("# Base Smoke 0.3.2\n\nSeparate Fast/gate optimization continuation. Fixed-set pass is not generalization.\n",encoding="utf-8");print(json.dumps(result,indent=2));return 0
if __name__=="__main__":raise SystemExit(main())
