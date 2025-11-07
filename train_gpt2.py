from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import tiktoken
from transformers import set_seed
import numpy

# ------------------------------------------

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = True
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # not really a 'bias', more of a mask, but following the original GPT-2 naming convention
        # block_size: maximum context length for predictions
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))

                            
    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embdding dimensionality (n_embd)
        # nh: number of heads, hs: head size (n_embd // n_head), C(number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so C=768 channels
        qkv = self.c_attn(x) # (B, T, 3 * C)
        q, k, v = qkv.split(C, dim=2) # each is (B, T, C)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, T, nh, hs) -> (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        # attention (materializes the large (T,T) matrix for all the quries and keys)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) # (B, nh, T, hs) @ (B, nh, hs, T) -> (B, nh, T, T)
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v # (B, nh T, T) @ (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # (B, T, C)
        y = self.c_proj(y) 
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x
# ------------------------------------------

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x)) # residual connection, pre-normalization version. reduce
        x = x + self.mlp(self.ln_2(x)) # map
        return x



@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257 # 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    bias: bool = False
    device: str = "mps"


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd, device=config.device),
            wpe = nn.Embedding(config.block_size, config.n_embd, device=config.device),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd, device=config.device),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            std = 0.02
            if hasattr(module, "NANOGPT_SCALE_INIT"):
                std *= (2 * self.config.n_layer) ** -0.5
                # 2 times comes from 'mlp' and 'attention' in block layer
            module.weight.data.normal_(mean=0.0, std=std)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()

    def forward(self, idx, targets=None):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward, model block size({T}) is exhausted."
        # forward the token and position embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T,)
        pos_emb = self.transformer.wpe(pos) # (T, n_embd)
        tok_emb = self.transformer.wte(idx) # (B, T, n_embd)
        x = tok_emb + pos_emb # (B, T, n_embd)
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x) # (B, T, n_embd)
        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = None
        if targets is not None:
            # token_id = targets[:, -1].unsqueeze(1)
            # b = torch.zeros(B, self.config.vocab_size, dtype=torch.long, device=idx.device)
            # b.scatter_(1, token_id, 1)
            # loss = F.cross_entropy(logits, b)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {} # default to empty dict
        # only dropout can be overridden see more notes below
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        config_args['bias'] = True # always True for GPT model checkpoints
        # we can override the dropout rate, if desired
        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

class DataLoaderLite:
    def __init__(self, batch_size, block_size):
        with open('input.txt', 'r') as f:
            self.text = f.read()
        self.enc = tiktoken.get_encoding("gpt2")
        self.tokens = torch.tensor(self.enc.encode(self.text)).to(device)
        self.batch_size = batch_size
        self.block_size = block_size
        self.tokens_per_batch = self.batch_size * self.block_size
        self.current_idx = 0
        print(f"> loaded : {len(self.tokens)} tokens")
        print(f"> 1 epoch : {self.tokens_per_batch} ")
        
    def next_batch(self):
        base = self.tokens[self.current_idx : self.current_idx + self.tokens_per_batch + 1]
        x = base[:-1].view(self.batch_size, self.block_size)
        y = base[1:].view(self.batch_size, self.block_size)
        self.current_idx += self.tokens_per_batch
        if (self.current_idx + 1 >= len(self.tokens)):
            self.current_idx = 0
        return x, y

#------------------------------------------

num_return_sequences = 5
max_length = 30

device = torch.accelerator.current_accelerator()
print(f"> accelerator device: {device}")

# data batch
dl = DataLoaderLite(batch_size=8, block_size=256)

# get logits
#model = GPT.from_pretrained('gpt2')
config = GPTConfig()
config.device = device
model = GPT(GPTConfig())
model.to(device)

# optimizer
optimzer = torch.optim.AdamW(model.parameters(), lr=3e-4)

# train
for i in range(50):
    optimzer.zero_grad()
    x, y = dl.next_batch()
    logits, loss = model(x, y)
    import code; code.interact(local=locals())
    loss.backward()
    optimzer.step()
    print(f"> step:{i}, loss: {loss.item()}")

import sys; sys.exit(0)

tokens = enc.encode("Hello, I'm a language model, ")
tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
tokens = tokens.repeat(num_return_sequences, 1)
x = tokens.to(device)

set_seed(42)
while x.size(1) < max_length:
    with torch.no_grad():
        logits = model(x) # (B, T, vocab_size)
        logits = logits[:, -1, :] # take last position (B, vocab_size)
        # get the probability of the next token
        probs = F.softmax(logits, dim=-1)
        # do top-k sampling of 50 (huggingface pipeline default)
        # topk_probs here becomes (5, 50)
        topk_probs, topk_indices = torch.topk(probs, k=50, dim=-1)
        # select top-k probabilities
        ix = torch.multinomial(topk_probs, num_samples=1)
        # gather the corresponding indices
        xcol = torch.gather(topk_indices, dim=-1, index=ix)
        # append to the sequence
        x = torch.cat([x, xcol], dim=1)

for i in range(num_return_sequences):
    tokens = x[i,:].tolist()
    print(f"> {enc.decode(tokens)}")