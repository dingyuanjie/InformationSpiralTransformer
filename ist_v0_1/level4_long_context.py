import argparse, json
from pathlib import Path
import torch
import torch.nn.functional as F
from long_context_test import set_seed
from marked_retrieval_level2 import make_batch
from model import InformationSpiralTransformer

STAGES = [
    (128, 64, 1500, 32),
    (256, 128, 500, 32),
    (512, 256, 500, 32),
    (512, 509, 500, 32),
    (1024, 1021, 300, 8),
    (2048, 2045, 300, 4),
]

@torch.no_grad()
def evaluate(model, length, batches, batch_size, device):
    model.eval(); correct=total=0; losses=[]
    for _ in range(batches):
        x,y,_=make_batch(batch_size,length,length-3,16,device)
        logits=model(x)[:,-1,:16]; losses.append(F.cross_entropy(logits,y).item())
        correct+=(logits.argmax(-1)==y).sum().item(); total+=len(y)
    return {"length":length,"accuracy":correct/total,"loss":sum(losses)/len(losses),"samples":total}

def main():
    p=argparse.ArgumentParser(description="IST Level 4 reproducible long-context run")
    p.add_argument("--encoding",default="rope",choices=["absolute","sinusoidal","rope","dynamic_rope"])
    p.add_argument("--seed",type=int,default=313); p.add_argument("--batch-size",type=int,default=32)
    p.add_argument("--eval-batch-size",type=int,default=8); p.add_argument("--eval-batches",type=int,default=20)
    p.add_argument("--output-root",default="experiments/level4")
    p.add_argument("--resume-checkpoint", default=None)
    args=p.parse_args()
    run_dir=Path(args.output_root)/f"{args.encoding}_seed{args.seed}"; run_dir.mkdir(parents=True,exist_ok=True)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); set_seed(args.seed)
    model=InformationSpiralTransformer(19,64,3,2048,args.encoding).to(device)
    start_stage = 0
    if args.resume_checkpoint:
        checkpoint = torch.load(args.resume_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        completed = checkpoint["stage"]
        for index, stage in enumerate(STAGES):
            if stage[0] == completed["length"]:
                start_stage = index + 1
        print(f"resumed={args.resume_checkpoint} start_stage={start_stage}", flush=True)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-3); stages=[]
    for length,needle_range,steps,stage_batch_size in STAGES[start_stage:]:
        for step in range(1,steps+1):
            batch_size = min(args.batch_size, stage_batch_size)
            model.train(); x,y,pos=make_batch(batch_size,length,needle_range,16,device)
            opt.zero_grad(set_to_none=True); logits=model(x)[...,:16]; rows=torch.arange(len(y),device=device)
            q=F.cross_entropy(logits[:,-1],y); local=F.cross_entropy(logits[rows,pos],y)
            loss=q+0.5*local+0.1*model.memory_diversity_loss(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        metric=evaluate(model,length,10,min(args.eval_batch_size,stage_batch_size),device); stages.append(metric); print("stage",metric,flush=True)
        torch.save({"model":model.state_dict(),"config":vars(args),"stage":metric},run_dir/f"checkpoint_{length}_{needle_range}.pt")
    tests=[evaluate(model,n,args.eval_batches,args.eval_batch_size,device) for n in (512,1024,2048)]
    payload={"config":vars(args),"device":str(device),"stages":stages,"tests":tests}
    (run_dir/"metrics.json").write_text(json.dumps(payload,indent=2),encoding="utf-8")
    print(json.dumps(tests,indent=2)); print(f"saved={run_dir}")
if __name__=="__main__": main()
