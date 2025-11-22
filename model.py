import torch
import torch.nn as nn
import torch.nn.functional as F
from config import PicoGPTConfig, GPTConfig
from torch.nn.parallel import DistributedDataParallel as DDP
import inspect
import transformer_engine.pytorch as te
from transformer_engine.common import recipe

class RoPE(nn.Module):
    def __init__(self, head_dim:int, max_seq_len:int = 2048, base: int=10000):
        super().__init__()
        # the rotation frequencies are precomputed and fixed
        # 1. Calculate the inverse frequencies (theta_i in the formula)
        # This creates the vector [1/base^(0/d), 1/base^(2/d), 1/base^(4/d), ...]
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq)

        # 2. Precompute the rotation angles (m * theta_i) for max sequence length
        t = torch.arange(max_seq_len).float()
        freqs = torch.einsum("i,j->ij", t, self.inv_freq) # shape (max_seq_len, head_dim//2)

        # 3. Convert to complex numbers for rotation (cos + i*sin)
        self.register_buffer("freqs_cis", torch.polar(torch.ones_like(freqs), freqs))

    def forward(self, x):
        # x is assumed to be the Q or K tensor, shape(B, H, S, D)

        # Use only the frequencies needed for the current sequence length
        seq_len = x.size(-2)
        freqs_cis = self.freqs_cis[:seq_len]
        
        # 1. Split the last dimension into pairs and view as complex numbers
        # (B, H , S, D) -> (B , H, S, D/2, 2) -> (B, H, S, D/2) (complex)
        x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))

        # 2. Apply the rotaiton via complex multiplication
        # Broadcasts freqs_cis (S, D/2) across Batch and Head dimensions
        x_rotated = x_complex * freqs_cis.unsqueeze(0).unsqueeze(0)

        # 3. Convert back to real tensor and flatten
        # (B, H, S, D/2) (complex) -> (B, H, S, D/2, 2) (real) -> (B, H, S, D)
        x_out = torch.view_as_real(x_rotated).flatten(3)

        return x_out.type_as(x)

class CausalSelfAttention(nn.Module):

    def __init__(self, config: PicoGPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        assert config.n_head % config.n_kv_head == 0

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.n_kv_head = config.n_kv_head
        self.head_size = config.n_embd // config.n_head
        self.n_kv_embd = config.n_kv_head * self.head_size

        if config.rope:
            self.rope = RoPE(self.head_size, max_seq_len=config.sequence_length)
            self.preprocess = lambda x: self.rope(x)
        else:
            self.preprocess = lambda x: x

        # query + key + value projections for all heads, but in a batch
        total_projection_dim = self.n_embd + self.n_kv_embd * 2
        self.c_attn = te.Linear(config.n_embd, total_projection_dim)
        # output projection
        self.c_proj = te.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1


                            
    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embdding dimensionality (n_embd)

        qkv = self.c_attn(x) # (B, T, total_projection_dim)

        q, k, v = qkv.split([C, self.n_kv_embd, self.n_kv_embd], dim=2) # each is (B, T, differ)
        k = k.view(B, T, self.n_kv_head, self.head_size).transpose(1, 2)  # (B, T, nh, hs) -> (B, nh, T, hs)
        v = v.view(B, T, self.n_kv_head, self.head_size).transpose(1, 2)
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)

        q = self.preprocess(q)
        k = self.preprocess(k)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True) # (B, nh, T, hs)

        y = y.transpose(1, 2).contiguous().view(B, T, C) # (B, T, C)
        y = self.c_proj(y) 
        return y


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = te.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj =te.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config, index):
        super().__init__()
        self.attention_layer = index < config.n_attn
        
        if self.attention_layer:
            self.ln_1 = nn.LayerNorm(config.n_embd)
            self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        if self.attention_layer:
            x = x + self.attn(self.ln_1(x)) # residual connection, pre-normalization version. reduce
        x = x + self.mlp(self.ln_2(x)) # map
        return x


class GPT(nn.Module):
    def __init__(self, config:PicoGPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd, device=config.device),
            h = nn.ModuleList([Block(config, i) for i in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd, device=config.device),
        ))

        if config.ape:
            self.transformer.add_module("wpe", nn.Embedding(config.sequence_length, config.n_embd, device=config.device))

        self.lm_head = te.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "NANOGPT_SCALE_INIT"):
                std *= (2 * self.config.n_layer) ** -0.5
                # 2 times comes from 'mlp' and 'attention' in block layer
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def configure_optimizers(self, weight_decay, learning_rate, device):
        # start with all of the candiate parameters 
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        # create optim groups, Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embedding decay, all biases and layernorms don't
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        no_decay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_no_decay_params = sum(p.numel() for p in no_decay_params)
        print(f"num decay params: {len(decay_params)}, {num_decay_params:,} params")
        print(f"num no decay params: {len(no_decay_params)}, {num_no_decay_params:,} params")
        # Create AdamW optimizer and use the fused version if available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device.startswith("cuda")
        print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8)
        return optimizer

    def forward(self, idx, targets=None):

        # Create the fp8 recipe (E4M3 is common for training)
        fp8_recipe = recipe.DelayedScaling(
            margin=0,
            interval=1, # calibrate scales every 1 iteration
            fp8_format=recipe.Format.E4M3,
        )

        # use fp8_autocast to enable FP8 for all TE layers
        with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
            # idx is of shape (B, T)
            B, T = idx.size()
            assert T <= self.config.sequence_length, f"Cannot forward, model block size({T}) is exhausted."

            # forward the token and position embeddings
            x = self.transformer.wte(idx) # (B, T, n_embd)
            if self.config.ape:
                pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T,)
                pos_emb = self.transformer.wpe(pos) # (T, n_embd)
                x = x + pos_emb

            # forward the blocks of the transformer
            for block in self.transformer.h:
                x = block(x)
            x = self.transformer.ln_f(x) # (B, T, n_embd)
            logits = self.lm_head(x) # (B, T, vocab_size)
        
        # loss calculation remains in higher precision
        loss = None
        if targets is not None:
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

def load_model(config:PicoGPTConfig):
    model = GPT(config)
    model.to(config.device)
    use_compile = True # torch.compile interferes with HellaSwag eval and Generation. TODO fix
    if use_compile:
        model = torch.compile(model)
    if config.ddp:
        model = DDP(model, device_ids=[config.process_rank])
    raw_model = model.module if config.ddp else model # always contains the "raw" unwrapped model
    return model, raw_model

    # write down the different with rank and local_rank on Notion