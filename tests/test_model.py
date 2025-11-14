import torch
import pytest
import sys
from unittest.mock import MagicMock

# Mock the transformer_engine module before it's imported by other modules
transformer_engine = MagicMock()
transformer_engine.pytorch = MagicMock()
transformer_engine.common = MagicMock()
transformer_engine.common.recipe = MagicMock()
sys.modules['transformer_engine'] = transformer_engine
sys.modules['transformer_engine.pytorch'] = transformer_engine.pytorch
sys.modules['transformer_engine.common'] = transformer_engine.common
sys.modules['transformer_engine.common.recipe'] = transformer_engine.common.recipe


from model import GPT, CausalSelfAttention, MLP, Block
from config import PicoGPTConfig
import torch.nn as nn

@pytest.fixture
def gpt_config():
    return PicoGPTConfig(
        n_layer=2,
        n_head=2,
        n_embd=128,
        sequence_length=256,
        vocab_size=50257,
        n_kv_head=2,
        device='cpu',
        execution_id='test'
    )

def test_causal_self_attention(gpt_config):
    attn = CausalSelfAttention(gpt_config)
    x = torch.randn(1, gpt_config.sequence_length, gpt_config.n_embd)
    y = attn(x)
    assert y.shape == x.shape

def test_mlp(gpt_config):
    mlp = MLP(gpt_config)
    x = torch.randn(1, 10, gpt_config.n_embd)
    y = mlp(x)
    assert y.shape == x.shape

def test_block(gpt_config):
    block = Block(gpt_config)
    x = torch.randn(1, gpt_config.sequence_length, gpt_config.n_embd)
    y = block(x)
    assert y.shape == x.shape

def test_gpt_forward(gpt_config):
    model = GPT(gpt_config)
    idx = torch.randint(0, gpt_config.vocab_size, (1, gpt_config.sequence_length))
    logits, loss = model(idx)
    assert logits.shape == (1, gpt_config.sequence_length, gpt_config.vocab_size)
    assert loss is None

def test_gpt_forward_with_targets(gpt_config):
    model = GPT(gpt_config)
    idx = torch.randint(0, gpt_config.vocab_size, (1, gpt_config.sequence_length))
    targets = torch.randint(0, gpt_config.vocab_size, (1, gpt_config.sequence_length))
    logits, loss = model(idx, targets)
    assert logits.shape == (1, gpt_config.sequence_length, gpt_config.vocab_size)
    assert loss is not None
    assert isinstance(loss, torch.Tensor)

def test_gpt_optimizer_configuration(gpt_config):
    model = GPT(gpt_config)
    optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=0.001, device='cpu')
    assert isinstance(optimizer, torch.optim.AdamW)
    assert len(optimizer.param_groups) == 2