"""Convenience entry point for inference latency/peak-memory measurements."""
import argparse
import json
import torch

from config import V05Config
from model import HybridIST
from run_level_a import evaluate

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/level_a.json")
parser.add_argument("--chunks", nargs="+", type=int, default=[2, 4, 8, 16, 32])
parser.add_argument("--samples", type=int, default=32)
args = parser.parse_args()
config = V05Config.from_json(args.config); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HybridIST(config).to(device).eval()
print(json.dumps([evaluate(model, config, 505, chunks, args.samples, device) for chunks in args.chunks], indent=2))
