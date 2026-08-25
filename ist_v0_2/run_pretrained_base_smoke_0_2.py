"""Base Smoke 0.2: fixed-set 1K overfit and Memory gradient diagnosis."""
from __future__ import annotations
import argparse,json,random
import torch
import torch.nn.functional as F
from experiment_utils import ROOT,atomic_json,atomic_torch,run_metadata,parameter_count
from pretrained_memory_adapter import FrozenPretrainedIST,load_qwen
from run_pretrained_base_smoke import MODEL_ID,candidate_ids,make_tokens,chunks,alter,set_seed

DISTANCE=1024;FIXED=32
def fixed_data(tokenizer,device,base_seed=93000000,split="train"):
    rows=[make_tokens(tokenizer,base_seed+i,split,DISTANCE) for i in range(FIXED)]
    return [(tokens.to(device),target,meta) for tokens,target,meta in rows]
def batch_of(rows,indices):
    return torch.stack([rows[i][0] for i in indices]),torch.tensor([rows[i][1] for i in indices],device=rows[0][0].device)
def grad_norm(module):
    values=[p.grad.detach().float().norm().square() for p in module.parameters() if p.grad is not None]
    return float(torch.stack(values).sum().sqrt()) if values else 0.0
def diagnostics(adapter):
    d=adapter.memory.last_diagnostics
    return {"router_distribution":d["router_distribution"].float().cpu().tolist(),"fast_write_rate":float(d["fast_write_rate"].cpu()),"slow_write_rate":float(d["slow_write_rate"].cpu()),"episodic_write_rate":float(d["episodic_write_rate"].cpu()),"retention_gate":float(d["retention_gate"].cpu()),"memory_scale":float(torch.tanh(adapter.memory_scale).detach())}
@torch.no_grad()
def score(adapter,labels,data,condition,batch=4):
    correct=[]
    for start in range(0,len(data),batch):
        ids,target=batch_of(data,list(range(start,min(start+batch,len(data)))));parts=list(ids.split(512,dim=1));state=None
        for part in parts[:-1]:_,state=adapter(part,state,detach_state=True)
        memory=alter(state,condition);intervention=condition if condition in ("zero_fast","zero_slow","zero_episodic") else "normal";logits,_=adapter(parts[-1],memory,intervention=intervention,detach_state=True);prediction=logits[:,-1,labels.to(ids.device)].argmax(-1);correct+=(prediction==target).int().cpu().tolist()
    adapter.clear_intervention();return sum(correct)/len(correct)
def train(adapter,tokenizer,labels,data,steps,batch,device,dtype,root):
    parameters=adapter.trainable_parameters()+[adapter.memory_scale];optimizer=torch.optim.AdamW(parameters,lr=2e-4);history=[];start=0;resume=root/"training_resume.pt"
    if resume.exists():
        saved=torch.load(resume,map_location=device,weights_only=False);adapter.memory.load_state_dict(saved["memory"]);adapter.memory_scale.data.copy_(saved["memory_scale"]);optimizer.load_state_dict(saved["optimizer"]);history=saved["history"];start=int(saved["step"]);print(f"resume step={start}",flush=True)
    stable=0
    for step in range(start+1,steps+1):
        rng=random.Random(94000000+step);indices=[rng.randrange(FIXED) for _ in range(batch)];ids,target=batch_of(data,indices);adapter.train();optimizer.zero_grad(set_to_none=True);state=None
        with torch.autocast(device_type="cuda",dtype=dtype):
            for part in ids.split(512,dim=1):logits,state=adapter(part,state)
            candidates=logits[:,-1,labels.to(device)];task=F.cross_entropy(candidates,target);teacher=adapter.last_base_logits[:,-1].float();student=logits[:,-1].float();distill=F.kl_div(F.log_softmax(student,dim=-1),F.softmax(teacher,dim=-1),reduction="batchmean");loss=task+.02*distill
        loss.backward();gradient={"fast":grad_norm(adapter.memory.fast_writer),"slow":grad_norm(adapter.memory.slow_candidate)+grad_norm(adapter.memory.slow_write_gate),"episodic":grad_norm(adapter.memory.episodic_key)+grad_norm(adapter.memory.episodic_value),"router":grad_norm(adapter.memory.router),"scale":float(adapter.memory_scale.grad.float().abs()) if adapter.memory_scale.grad is not None else 0.0};torch.nn.utils.clip_grad_norm_(parameters,1);optimizer.step()
        if step==1 or step%25==0:
            row={"step":step,"loss":float(loss.detach()),"task_loss":float(task.detach()),"distill_loss":float(distill.detach()),"batch_accuracy":float((candidates.argmax(-1)==target).float().mean()),"gradient_norms":gradient,"diagnostics":diagnostics(adapter)}
            if step%50==0:
                adapter.eval();row["fixed_normal"]=score(adapter,labels,data,"normal",batch);row["fixed_zero_memory"]=score(adapter,labels,data,"zero_memory",batch);row["fixed_reset_memory"]=score(adapter,labels,data,"reset_memory",batch);gap=row["fixed_normal"]-max(row["fixed_zero_memory"],row["fixed_reset_memory"]);stable=stable+1 if row["fixed_normal"]>=.95 and gap>=.5 else 0;row["causal_gap"]=gap;row["stable_checks"]=stable
            history.append(row);print(json.dumps(row),flush=True);atomic_torch(resume,{"memory":adapter.memory.state_dict(),"memory_scale":adapter.memory_scale.detach(),"optimizer":optimizer.state_dict(),"history":history,"step":step})
            if stable>=2:print("OVERFIT_CAUSAL_GATE_PASS",flush=True);break
    return history,optimizer,history[-1]["step"]
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--model-id",default=MODEL_ID);p.add_argument("--steps",type=int,default=600);p.add_argument("--batch",type=int,default=4);p.add_argument("--output",default="experiments/pretrained_base/base_smoke_0_2/formal");p.add_argument("--dry-run",action="store_true");p.add_argument("--local-files-only",action="store_true");args=p.parse_args();protocol={"model_id":args.model_id,"task":"fixed-set 1K overfit","fixed_examples":FIXED,"steps":args.steps,"batch":args.batch,"freeze_backbone":True,"identity_preserving":True,"gate":{"fixed_normal":.95,"normal_minus_zero_or_reset":.5,"consecutive_checks":2},"not_a_generalization_result":True}
    if args.dry_run:print(json.dumps(protocol,indent=2));return 0
    if not torch.cuda.is_available():raise RuntimeError("CUDA required")
    device=torch.device("cuda");dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16;root=ROOT/args.output;root.mkdir(parents=True,exist_ok=True);set_seed(950001);tokenizer,backbone=load_qwen(args.model_id,dtype,device,args.local_files_only);labels=candidate_ids(tokenizer);adapter=FrozenPretrainedIST(backbone,identity_preserving=True).to(device=device,dtype=dtype);data=fixed_data(tokenizer,device);history,optimizer,completed=train(adapter,tokenizer,labels,data,args.steps,args.batch,device,dtype,root);conditions=("normal","zero_memory","reset_memory","roll_memory","zero_fast","zero_slow","zero_episodic");final={c:score(adapter,labels,data,c,args.batch) for c in conditions};held=fixed_data(tokenizer,device,96000000,"held_out");heldout={c:score(adapter,labels,held,c,args.batch) for c in conditions};gap=final["normal"]-max(final["zero_memory"],final["reset_memory"]);passed=final["normal"]>=.95 and gap>=.5;revision=getattr(backbone.config,"_commit_hash",None);protocol["resolved_revision"]=revision;result={"status":"complete","overfit_causal_gate_passed":passed,"completed_steps":completed,"fixed_train":final,"heldout_diagnostic":heldout,"causal_gap":gap,"final_diagnostics":diagnostics(adapter),"history":history,"protocol":protocol,"backbone_parameters":parameter_count(backbone),"trainable_parameters":sum(p.numel() for p in adapter.trainable_parameters())+1};atomic_json(root/"config.json",protocol);atomic_json(root/"run_metadata.json",run_metadata(device,950001));atomic_torch(root/"memory_checkpoint.pt",{"memory":adapter.memory.state_dict(),"memory_scale":adapter.memory_scale.detach(),"optimizer":optimizer.state_dict(),"history":history,"revision":revision});atomic_json(root/"raw_results.json",result);atomic_json(root/"result.json",result);(root/"ANALYSIS.md").write_text("# Base Smoke 0.2\n\nThis is an architecture learnability diagnostic, not a held-out performance claim.\n",encoding="utf-8");print(json.dumps(result,indent=2));return 0
if __name__=="__main__":raise SystemExit(main())
