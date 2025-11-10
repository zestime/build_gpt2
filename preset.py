from config import GPTConfig


GPT2Preset = GPTConfig(
    preset="gpt2",
    block_size=1024,
    vocab_size=50257,
    n_layer=12,
    n_head=12,
    n_embd=768,
    bias=False,
)

GPT2XLConfig = GPTConfig(
    preset="gpt2-xl",
    block_size=1024,
    vocab_size=50257,
    n_layer=24,
    n_head=16,
    n_embd=1024,
    bias=False,
)
