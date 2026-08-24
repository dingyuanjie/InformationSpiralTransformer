"""0.5B Base smoke: same-checkpoint Base vs frozen-backbone IST Memory."""
from __future__ import annotations
import argparse,json,random
import torch
import torch.nn.functional as F
from experiment_utils import ROOT,atomic_json,atomic_torch,run_metadata,parameter_count
from natural_language_data import generate_nl1,DISTRACTORS
from pretrained_memory_adapter import FrozenPretrainedIST,load_qwen

MODEL_ID="Qwen/Qwen2.5-0.5B";CHUNK=512
def set_seed(seed):torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);random.seed(seed)
def candidate_ids(tokenizer):
    rows=[]
    for label in "ABCD":
        ids=tokenizer.encode(" "+label,add_special_tokens=False)
        if len(ids)!=1:raise RuntimeError(f"answer label {label!r} is not one token: {ids}")
        rows.append(ids[0])
    return torch.tensor(rows)
def make_tokens(tokenizer,seed,split,distance):
    example=generate_nl1(seed,split,max(512,distance),512,option_count=4);m=example.metadata
    fact=tokenizer.encode(m["fact"]+"\n",add_special_tokens=False);query=tokenizer.encode("\n"+m["query"],add_special_tokens=False)
    rng=random.Random(seed);filler=[]
    while len(filler)<distance-len(fact)-len(query):filler+=tokenizer.encode(rng.choice(DISTRACTORS)+" ",add_special_tokens=False)
    ids=(fact+filler[:max(0,distance-len(fact)-len(query))]+query)[-distance:]
    if len(ids)!=distance:raise RuntimeError(f"constructed {len(ids)} tokens, expected {distance}")
    return torch.tensor(ids),m["answer_index"],m
def chunks(tokens):return list(tokens.split(CHUNK))
def alter(state,condition):
    if state is None:return None
    if condition=="reset_memory":return None
    result={k:v.clone() if torch.is_tensor(v) else v for k,v in state.items()}
    if condition=="zero_memory":
        for key in ("fast","slow","episodic_keys","episodic_values"):result[key].zero_()
    elif condition=="roll_memory":
        for key in ("fast","slow","episodic_keys","episodic_values"):result[key]=torch.roll(result[key],1,1)
    return result
def train(adapter,tokenizer,labels,steps,distance,device,dtype,seed):
    optimizer=torch.optim.AdamW(adapter.trainable_parameters(),lr=2e-4);history=[]
    for step in range(1,steps+1):
        set_seed(87000000+seed*1000+step);ids,target,_=make_tokens(tokenizer,87000000+seed*1000+step,"train",distance);state=None;adapter.train();optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda",dtype=dtype):
            for part in chunks(ids.to(device)):logits,state=adapter(part[None],state)
            scores=logits[0,-1,labels.to(device)];loss=F.cross_entropy(scores[None],torch.tensor([target],device=device))
        loss.backward();torch.nn.utils.clip_grad_norm_(adapter.trainable_parameters(),1);optimizer.step()
        if step==1 or step%10==0:history.append({"step":step,"loss":float(loss.detach()),"correct":int(scores.argmax()==target)});print(f"step={step} loss={float(loss.detach()):.4f} correct={int(scores.argmax()==target)}",flush=True)
    return history,optimizer
@torch.no_grad()
def evaluate(backbone,adapter,tokenizer,labels,distances,samples,device):
    rows=[];conditions=("base","normal","zero_memory","reset_memory","roll_memory","zero_fast","zero_slow","zero_episodic")
    for distance in distances:
      for condition in conditions:
        correct=[]
        for i in range(samples):
            ids,target,_=make_tokens(tokenizer,88000000+distance*10+i,"held_out",distance);ids=ids.to(device)
            if condition=="base":logits=backbone(ids[None],use_cache=False).logits;prediction=logits[0,-1,labels.to(device)].argmax()
            else:
                state=None
                for part in chunks(ids)[:-1]:_,state=adapter(part[None],state,detach_state=True)
                memory=alter(state,condition);intervention=condition if condition in ("zero_fast","zero_slow","zero_episodic") else "normal"
                logits,_=adapter(chunks(ids)[-1][None],memory,intervention=intervention,detach_state=True);prediction=logits[0,-1,labels.to(device)].argmax()
            correct.append(int(prediction==target))
        rows.append({"distance":distance,"condition":condition,"correct":sum(correct),"samples":len(correct),"accuracy":sum(correct)/len(correct)});print(f"distance={distance} condition={condition} accuracy={rows[-1]['accuracy']:.2%}",flush=True)
    adapter.clear_intervention();return rows
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--model-id",default=MODEL_ID);p.add_argument("--steps",type=int,default=50);p.add_argument("--distances",nargs="+",type=int,default=[512,1024,2048]);p.add_argument("--samples",type=int,default=16);p.add_argument("--output",default="experiments/pretrained_base/base_smoke/formal");p.add_argument("--dry-run",action="store_true");p.add_argument("--local-files-only",action="store_true");args=p.parse_args()
    protocol={"model_id":args.model_id,"freeze_backbone":True,"trainable":["hierarchical_memory","read","write","fusion"],"steps":args.steps,"distances":args.distances,"samples":args.samples,"chunk_size":CHUNK,"conditions":["base","normal","zero_memory","reset_memory","roll_memory","zero_fast","zero_slow","zero_episodic"]}
    if args.dry_run:print(json.dumps(protocol,indent=2));return 0
    if not torch.cuda.is_available():raise RuntimeError("0.5B smoke requires CUDA")
    device=torch.device("cuda");dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16;root=ROOT/args.output;root.mkdir(parents=True,exist_ok=True);set_seed(890001)
    tokenizer,backbone=load_qwen(args.model_id,dtype,device,args.local_files_only);labels=candidate_ids(tokenizer);adapter=FrozenPretrainedIST(backbone).to(device=device,dtype=dtype)
    phase=root/"training_complete.pt"
    if phase.exists():
        saved=torch.load(phase,map_location=device,weights_only=False);adapter.memory.load_state_dict(saved["memory"]);history=saved["history"];optimizer=torch.optim.AdamW(adapter.trainable_parameters(),lr=2e-4);optimizer.load_state_dict(saved["optimizer"]);print("Frozen-Memory training already complete; resuming evaluation.",flush=True)
    else:
        history,optimizer=train(adapter,tokenizer,labels,args.steps,min(args.distances),device,dtype,890001);atomic_torch(phase,{"memory":adapter.memory.state_dict(),"optimizer":optimizer.state_dict(),"history":history})
    rows=evaluate(backbone,adapter,tokenizer,labels,args.distances,args.samples,device)
    revision=getattr(backbone.config,"_commit_hash",None);protocol["resolved_revision"]=revision;atomic_json(root/"config.json",protocol);atomic_json(root/"run_metadata.json",run_metadata(device,890001));atomic_torch(root/"memory_checkpoint.pt",{"memory":adapter.memory.state_dict(),"optimizer":optimizer.state_dict(),"model_id":args.model_id,"revision":revision,"history":history});result={"status":"complete","protocol":protocol,"backbone_parameters":parameter_count(backbone),"trainable_memory_parameters":sum(p.numel() for p in adapter.trainable_parameters()),"history":history,"summary":rows};atomic_json(root/"raw_results.json",result);atomic_json(root/"result.json",result);(root/"ANALYSIS.md").write_text("# Pretrained Base Smoke\n\nCompare Base, IST normal, and causal controls. Smoke results do not authorize a scaling claim.\n",encoding="utf-8");print(json.dumps(result,indent=2));return 0
if __name__=="__main__":raise SystemExit(main())
