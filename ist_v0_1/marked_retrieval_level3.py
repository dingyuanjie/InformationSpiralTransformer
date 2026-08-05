import argparse
import json

import torch
import torch.nn.functional as F

from long_context_test import set_seed
from marked_retrieval_level2 import make_batch, evaluate
from model import InformationSpiralTransformer


STAGES = [
    {"name": "0-64", "length": 128, "needle_range": 64, "steps": 1500},
    {"name": "0-128", "length": 256, "needle_range": 128, "steps": 500},
    {"name": "0-256", "length": 512, "needle_range": 256, "steps": 500},
    {"name": "0-512", "length": 512, "needle_range": 509, "steps": 500},
]


def main():
    parser = argparse.ArgumentParser(description="Level 3 distance curriculum")
    parser.add_argument("--steps-per-stage", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--validation-batches", type=int, default=20)
    parser.add_argument("--seed", type=int, default=313)
    parser.add_argument("--output", default="experiments/results/level3_results.json")
    args = parser.parse_args(); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    model = InformationSpiralTransformer(19, 64, 3, 512, "rope").to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    results = []
    for stage in STAGES:
        print(f"\nStage {stage['name']} length={stage['length']}")
        stage_steps = min(args.steps_per_stage, stage["steps"])
        for step in range(1, stage_steps + 1):
            model.train(); tokens, targets, positions = make_batch(
                args.batch_size, stage["length"], stage["needle_range"], 16, device)
            optimizer.zero_grad(set_to_none=True); logits = model(tokens)[..., :16]
            rows = torch.arange(args.batch_size, device=device)
            query_loss = F.cross_entropy(logits[:, -1], targets)
            local_loss = F.cross_entropy(logits[rows, positions], targets)
            loss = query_loss + 0.5 * local_loss + 0.1 * model.memory_diversity_loss()
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            if step == 1 or step % 100 == 0:
                accuracy = (logits[:, -1].argmax(-1) == targets).float().mean().item()
                print(f"step={step:03d} loss={query_loss.item():.4f} acc={accuracy:.2%}")
        eval_args = argparse.Namespace(batch_size=args.batch_size,
            validation_batches=args.validation_batches, length=stage["length"],
            needle_range=stage["needle_range"], vocab_size=16)
        validation = evaluate(model, eval_args, device)
        passed = validation["query_accuracy"] >= 0.90
        results.append({"stage": stage, "validation": validation, "passed": passed})
        print(f"validation={validation} {'PASS' if passed else 'FAIL'}")
        if not passed:
            break
    overall = len(results) == len(STAGES) and all(item["passed"] for item in results)
    with open(args.output, "w", encoding="utf-8") as output:
        json.dump({"config": vars(args), "stages": results, "passed": overall}, output, indent=2)
    print("LEVEL3_PASS" if overall else "LEVEL3_FAIL")


if __name__ == "__main__": main()
