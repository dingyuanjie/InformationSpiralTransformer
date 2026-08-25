"""Base Smoke 0.3: persistence-only fixed-set causal overfit diagnosis."""
from __future__ import annotations
import argparse,json
import torch
from experiment_utils import ROOT,atomic_json,atomic_torch,run_metadata,parameter_count
from pretrained_memory_adapter import FrozenPretrainedIST,load_qwen
from run_pretrained_base_smoke import MODEL_ID,candidate_ids,make_tokens
from run_pretrained_base_smoke_0_2 import fixed_data,train,score,diagnostics

@torch.no_grad()
def invariants(backbone,adapter,tokenizer,device):
    ids,_,_=make_tokens(tokenizer,970001,"held_out",512);ids=ids.to(device)
    base=backbone(ids[None],use_cache=False).logits[:,-1:]
    no_history,_=adapter(ids[None],None)
    first_delta=float((base-no_history).abs().max().cpu())
    prior,_,_=make_tokens(tokenizer,970002,"held_out",512);_,state=adapter(prior.to(device)[None],None,detach_state=True)
    reset,_=adapter(ids[None],None)
    reset_delta=float((base-reset).abs().max().cpu())
    adapter.memory_scale.fill_(10)
    normal,_=adapter(ids[None],state)
    historical_signal=float((normal-reset).abs().max().cpu())
    adapter.memory_scale.zero_()
    return {"no_history_vs_base_max_delta":first_delta,"reset_vs_base_max_delta":reset_delta,
            "random_initial_historical_signal":historical_signal}
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--model-id",default=MODEL_ID);p.add_argument("--steps",type=int,default=600);p.add_argument("--batch",type=int,default=4);p.add_argument("--output",default="experiments/pretrained_base/base_smoke_0_3/formal");p.add_argument("--dry-run",action="store_true");p.add_argument("--local-files-only",action="store_true");args=p.parse_args();protocol={"model_id":args.model_id,"task":"persistence-only fixed-set 1K overfit","fixed_examples":32,"steps":args.steps,"batch":args.batch,"freeze_backbone":True,"adapter":"hidden + zero_initialized_scale * (memory(history)-memory(reset))","gate":{"fixed_normal":.95,"normal_minus_zero_or_reset":.5,"consecutive_checks":2},"hard_invariants":["no-history equals Base","reset equals Base"],"not_a_generalization_result":True}
    if args.dry_run:print(json.dumps(protocol,indent=2));return 0
    if not torch.cuda.is_available():raise RuntimeError("CUDA required")
    device=torch.device("cuda");dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16;root=ROOT/args.output;root.mkdir(parents=True,exist_ok=True);tokenizer,backbone=load_qwen(args.model_id,dtype,device,args.local_files_only);labels=candidate_ids(tokenizer);adapter=FrozenPretrainedIST(backbone,identity_preserving=True,persistence_only=True).to(device=device,dtype=dtype);checks=invariants(backbone,adapter,tokenizer,device)
    if checks["no_history_vs_base_max_delta"]!=0 or checks["reset_vs_base_max_delta"]!=0:raise RuntimeError(f"persistence-only invariant failed: {checks}")
    data=fixed_data(tokenizer,device);history,optimizer,completed=train(adapter,tokenizer,labels,data,args.steps,args.batch,device,dtype,root);conditions=("normal","zero_memory","reset_memory","roll_memory","zero_fast","zero_slow","zero_episodic");fixed={c:score(adapter,labels,data,c,args.batch) for c in conditions};held=fixed_data(tokenizer,device,98000000,"held_out");heldout={c:score(adapter,labels,held,c,args.batch) for c in conditions};gap=fixed["normal"]-max(fixed["zero_memory"],fixed["reset_memory"]);passed=fixed["normal"]>=.95 and gap>=.5;revision=getattr(backbone.config,"_commit_hash",None);protocol["resolved_revision"]=revision;result={"status":"complete","persistence_causal_gate_passed":passed,"invariants":checks,"completed_steps":completed,"fixed_train":fixed,"heldout_diagnostic":heldout,"causal_gap":gap,"final_diagnostics":diagnostics(adapter),"history":history,"protocol":protocol,"backbone_parameters":parameter_count(backbone),"trainable_parameters":sum(p.numel() for p in adapter.trainable_parameters())+1};atomic_json(root/"config.json",protocol);atomic_json(root/"run_metadata.json",run_metadata(device,990001));atomic_torch(root/"memory_checkpoint.pt",{"memory":adapter.memory.state_dict(),"memory_scale":adapter.memory_scale.detach(),"optimizer":optimizer.state_dict(),"history":history,"revision":revision});atomic_json(root/"raw_results.json",result);atomic_json(root/"result.json",result);(root/"ANALYSIS.md").write_text("# Base Smoke 0.3\n\nPersistence-only diagnostic. A training-set pass is causal learnability, not generalization.\n",encoding="utf-8");print(json.dumps(result,indent=2));return 0
if __name__=="__main__":raise SystemExit(main())
