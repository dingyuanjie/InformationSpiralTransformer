import json, statistics, time
from types import SimpleNamespace
import torch
import torch.nn.functional as F
from length_curriculum import LengthCurriculum, make_random_needle_batch, distance_bucket
from long_context_test import set_seed
from model import InformationSpiralTransformer

ENCODINGS = ["absolute", "sinusoidal", "rope", "dynamic_rope"]
SEEDS = [313, 42]
TEST_LENGTHS = [128, 256, 512, 1024, 2048]

def train_model(model, seed, device, steps=300):
    set_seed(seed); opt=torch.optim.AdamW(model.parameters(), lr=1e-3)
    schedule=LengthCurriculum("curriculum", 32, 512); model.train()
    for step in range(1, steps+1):
        length=schedule.sample(step, steps)
        x,y,_=make_random_needle_batch(64,length,16,device)
        opt.zero_grad(set_to_none=True); logits=model(x)[:,-1,:16]
        loss=F.cross_entropy(logits,y)+0.1*model.memory_diversity_loss()
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
    return loss.item()

@torch.no_grad()
def evaluate(model, device):
    model.eval(); buckets={}; lengths={}
    for length in TEST_LENGTHS:
        correct=total=0
        for _ in range(5):
            x,y,d=make_random_needle_batch(32,length,16,device); pred=model(x)[:,-1,:16].argmax(-1)
            correct+=(pred==y).sum().item(); total+=len(y)
            for i in range(len(y)):
                key=distance_bucket(int(d[i])); entry=buckets.setdefault(key,[0,0]); entry[0]+=int(pred[i]==y[i]); entry[1]+=1
        lengths[str(length)]=correct/total
    return lengths,{k:{"accuracy":v[0]/v[1],"samples":v[1]} for k,v in buckets.items()}

def main():
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); results=[]
    for encoding in ENCODINGS:
        runs=[]
        for seed in SEEDS:
            print(f"training {encoding} seed={seed}"); set_seed(seed)
            # 16 data tokens + [MASK] + [NEEDLE] + [QUERY].
            model=InformationSpiralTransformer(19,64,2,max(TEST_LENGTHS),encoding).to(device)
            started=time.perf_counter(); loss=train_model(model,seed,device)
            lengths,buckets=evaluate(model,device); score=statistics.mean(lengths.values())
            runs.append({"seed":seed,"score":score,"final_loss":loss,"seconds":time.perf_counter()-started,"lengths":lengths,"distance_buckets":buckets})
            print(f"score={score:.2%} lengths={lengths}"); del model; torch.cuda.empty_cache() if device.type=="cuda" else None
        scores=[r["score"] for r in runs]
        results.append({"position_encoding":encoding,"mean_accuracy":statistics.mean(scores),"std_accuracy":statistics.stdev(scores),"runs":runs})
    results.sort(key=lambda x:(x["mean_accuracy"],-x["std_accuracy"]),reverse=True)
    with open("experiments/results/curriculum_screening.json","w",encoding="utf-8") as f: json.dump({"top_two":[r["position_encoding"] for r in results[:2]],"results":results},f,indent=2)
    print("ranking:",[(r["position_encoding"],r["mean_accuracy"],r["std_accuracy"]) for r in results])

if __name__=="__main__": main()
