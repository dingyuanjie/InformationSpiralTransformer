"""Base Smoke 0.1: identity-preserving frozen Qwen + IST Memory."""
from __future__ import annotations
import argparse,json,random
import torch
import torch.nn.functional as F
from experiment_utils import ROOT,atomic_json,atomic_torch,run_metadata,parameter_count
from pretrained_memory_adapter import FrozenPretrainedIST,load_qwen
from run_pretrained_base_smoke import MODEL_ID,CHUNK,candidate_ids,make_tokens,chunks,alter,set_seed

def train(adapter,tokenizer,labels,steps,device,dtype,seed,root):
    optimizer=torch.optim.AdamW(adapter.trainable_parameters()+[adapter.memory_scale],lr=1e-4);history=[]
    resume=root/"training_resume.pt";start=0
    if resume.exists():
        saved=torch.load(resume,map_location=device,weights_only=False);adapter.memory.load_state_dict(saved["memory"]);adapter.memory_scale.data.copy_(saved["memory_scale"]);optimizer.load_state_dict(saved["optimizer"]);history=saved["history"];start=int(saved["step"]);print(f"resume training step={start}",flush=True)
    for step in range(start+1,steps+1):
        set_seed(90000000+seed*1000+step);distance=512 if step<=steps//2 else 1024;ids,target,_=make_tokens(tokenizer,90000000+seed*1000+step,"train",distance);state=None;adapter.train();optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda",dtype=dtype):
            for part in chunks(ids.to(device)):
                logits,state=adapter(part[None],state)
            candidate=logits[0,-1,labels.to(device)];teacher=adapter.last_base_logits[0,-1].float();student=logits[0,-1].float()
            task=F.cross_entropy(candidate[None],torch.tensor([target],device=device));distill=F.kl_div(F.log_softmax(student,dim=-1),F.softmax(teacher,dim=-1),reduction="sum");loss=task+.05*distill
        loss.backward();torch.nn.utils.clip_grad_norm_(adapter.trainable_parameters()+[adapter.memory_scale],1);optimizer.step()
        if step==1 or step%10==0:
            row={"step":step,"distance":distance,"loss":float(loss.detach()),"task_loss":float(task.detach()),"distill_loss":float(distill.detach()),"correct":int(candidate.argmax()==target),"memory_scale":float(torch.tanh(adapter.memory_scale).detach())};history.append(row);print(f"step={step} distance={distance} task={row['task_loss']:.4f} distill={row['distill_loss']:.4f} scale={row['memory_scale']:.6f} correct={row['correct']}",flush=True);atomic_torch(resume,{"memory":adapter.memory.state_dict(),"memory_scale":adapter.memory_scale.detach(),"optimizer":optimizer.state_dict(),"history":history,"step":step})
    return history,optimizer
@torch.no_grad()
def identity_check(backbone,adapter,tokenizer,device):
    ids,_,_=make_tokens(tokenizer,901,"held_out",512);ids=ids.to(device);base=backbone(ids[None],use_cache=False).logits[:,-1:];adapted,_=adapter(ids[None]);return float((base-adapted).abs().max().cpu())
@torch.no_grad()
def evaluate(backbone,adapter,tokenizer,labels,distances,samples,device):
    rows=[];conditions=("base","normal","zero_memory","reset_memory","roll_memory","zero_fast","zero_slow","zero_episodic")
    for distance in distances:
      for condition in conditions:
        correct=[];agreements=[]
        for i in range(samples):
            ids,target,_=make_tokens(tokenizer,91000000+distance*10+i,"held_out",distance);ids=ids.to(device);base_logits=backbone(ids[None],use_cache=False).logits[:,-1];base_prediction=base_logits[0,labels.to(device)].argmax()
            if condition=="base":prediction=base_prediction
            else:
                parts=chunks(ids);state=None
                for part in parts[:-1]:_,state=adapter(part[None],state,detach_state=True)
                memory=alter(state,condition);intervention=condition if condition in ("zero_fast","zero_slow","zero_episodic") else "normal";logits,_=adapter(parts[-1][None],memory,intervention=intervention,detach_state=True);prediction=logits[0,-1,labels.to(device)].argmax()
            correct.append(int(prediction==target));agreements.append(int(prediction==base_prediction))
        rows.append({"distance":distance,"condition":condition,"correct":sum(correct),"samples":samples,"accuracy":sum(correct)/samples,"base_agreement":sum(agreements)/samples});print(f"distance={distance} condition={condition} accuracy={rows[-1]['accuracy']:.2%} base_agreement={rows[-1]['base_agreement']:.2%}",flush=True)
    adapter.clear_intervention();return rows
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--model-id",default=MODEL_ID);p.add_argument("--steps",type=int,default=200);p.add_argument("--distances",nargs="+",type=int,default=[512,1024,2048]);p.add_argument("--samples",type=int,default=32);p.add_argument("--output",default="experiments/pretrained_base/base_smoke_0_1/formal");p.add_argument("--dry-run",action="store_true");p.add_argument("--local-files-only",action="store_true");args=p.parse_args();protocol={"model_id":args.model_id,"adapter":"zero-initialized residual hierarchical IST","freeze_backbone":True,"steps":args.steps,"curriculum":[512,1024],"distances":args.distances,"samples":args.samples,"distillation_weight":.05,"conditions":["base","normal","zero_memory","reset_memory","roll_memory","zero_fast","zero_slow","zero_episodic"]}
    if args.dry_run:print(json.dumps(protocol,indent=2));return 0
    if not torch.cuda.is_available():raise RuntimeError("CUDA required")
    device=torch.device("cuda");dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16;root=ROOT/args.output;root.mkdir(parents=True,exist_ok=True);tokenizer,backbone=load_qwen(args.model_id,dtype,device,args.local_files_only);labels=candidate_ids(tokenizer);adapter=FrozenPretrainedIST(backbone,identity_preserving=True).to(device=device,dtype=dtype);initial_delta=identity_check(backbone,adapter,tokenizer,device)
    if initial_delta!=0:raise RuntimeError(f"identity invariant failed: max logit delta={initial_delta}")
    history,optimizer=train(adapter,tokenizer,labels,args.steps,device,dtype,920001,root);rows=evaluate(backbone,adapter,tokenizer,labels,args.distances,args.samples,device);revision=getattr(backbone.config,"_commit_hash",None);protocol["resolved_revision"]=revision;result={"status":"complete","identity_max_logit_delta":initial_delta,"final_memory_scale":float(torch.tanh(adapter.memory_scale).detach()),"backbone_parameters":parameter_count(backbone),"trainable_parameters":sum(p.numel() for p in adapter.trainable_parameters())+1,"protocol":protocol,"history":history,"summary":rows};atomic_json(root/"config.json",protocol);atomic_json(root/"run_metadata.json",run_metadata(device,920001));atomic_torch(root/"memory_checkpoint.pt",{"memory":adapter.memory.state_dict(),"memory_scale":adapter.memory_scale.detach(),"optimizer":optimizer.state_dict(),"history":history,"revision":revision});atomic_json(root/"raw_results.json",result);atomic_json(root/"result.json",result);(root/"ANALYSIS.md").write_text("# Base Smoke 0.1\n\nIdentity invariant passed. Interpret preservation, long-range accuracy, and causal interventions jointly.\n",encoding="utf-8");print(json.dumps(result,indent=2));return 0
if __name__=="__main__":raise SystemExit(main())
