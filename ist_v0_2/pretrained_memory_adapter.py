"""Frozen causal-LM backbone with trainable IST hierarchical Memory."""
from __future__ import annotations
import torch
import torch.nn as nn
from config import HierarchicalMemoryConfig
from hierarchical_memory import HierarchicalMemory

class FrozenPretrainedIST(nn.Module):
    def __init__(self, backbone, memory_config=None, identity_preserving=True,
                 persistence_only=False):
        super().__init__();self.backbone=backbone
        hidden=int(backbone.config.hidden_size);config=memory_config or HierarchicalMemoryConfig()
        self.memory=HierarchicalMemory(hidden,config);self.memory_arch="pretrained_hierarchical_v0_2"
        self.identity_preserving=identity_preserving
        self.persistence_only=persistence_only
        self.memory_scale=nn.Parameter(torch.zeros(()))
        self.last_base_logits=None
        self.last_adapted_hidden=None
        for parameter in self.backbone.parameters():parameter.requires_grad_(False)
        self.backbone.eval()
    def train(self,mode=True):
        super().train(mode);self.backbone.eval();return self
    def forward(self,input_ids,state=None,intervention="normal",detach_state=False):
        self.memory.intervention=intervention
        with torch.no_grad():
            core=self.backbone.model(input_ids=input_ids,use_cache=False,return_dict=True)
            hidden=core.last_hidden_state.detach()
            self.last_base_logits=self.backbone.get_output_embeddings()(hidden[:,-1:]).detach()
        state,feature=self.memory(hidden,state)
        if self.persistence_only:
            historical_diagnostics=self.memory.last_diagnostics
            _,local_feature=self.memory(hidden,None)
            self.memory.last_diagnostics=historical_diagnostics
            delta=feature-local_feature
        else:
            delta=feature-hidden
        scale=torch.tanh(self.memory_scale.float()).to(hidden.dtype)
        adapted=(hidden+scale*delta) if self.identity_preserving else feature
        self.last_adapted_hidden=adapted[:,-1]
        logits=self.backbone.get_output_embeddings()(adapted[:,-1:])
        if detach_state:state=self.memory.detach_state(state)
        return logits,state
    def trainable_parameters(self):return [p for p in self.memory.parameters() if p.requires_grad]
    def clear_intervention(self):self.memory.intervention="normal"

def load_qwen(model_id="Qwen/Qwen2.5-0.5B",dtype=torch.bfloat16,device="cuda",local_files_only=False):
    try:from transformers import AutoModelForCausalLM,AutoTokenizer
    except ImportError as error:raise RuntimeError("Install requirements first: pip install -r requirements.txt") from error
    tokenizer=AutoTokenizer.from_pretrained(model_id,use_fast=True,local_files_only=local_files_only)
    backbone=AutoModelForCausalLM.from_pretrained(model_id,dtype=dtype,attn_implementation="sdpa",local_files_only=local_files_only).to(device)
    return tokenizer,backbone
