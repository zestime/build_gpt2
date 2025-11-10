from dataclasses import dataclass

@dataclass
class GPTConfig:
    preset: str = "gpt2"
    block_size: int = 1024
    vocab_size: int = 50257 # 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    bias: bool = False

@dataclass
class OptimizerConfig:
    max_learning_rate: float = 6e-4
    min_learning_rate: float = max_learning_rate * 0.1
    warmup_steps: int = 715
    max_steps: int = 19073
    weight_decay: float = 0.01
    save_checkpoint: int = 1000
    save_optimizer: bool = True

@dataclass
class TrainConfig:
    max_learning_rate: float = 6e-4
    min_learning_rate: float = max_learning_rate * 0.1
    warmup_steps: int = 715
    max_steps: int = 19073
    weight_decay: float = 0.01
    save_checkpoint: int = 1000
    save_optimizer: bool = True
    device: str = "cuda"