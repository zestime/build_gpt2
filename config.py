from functools import cached_property
from src.utils.utils import make_execution_id
from typing import Any, Type
from dataclasses import dataclass, is_dataclass, fields
import dataclasses
import inspect
import torch


def is_namedtuple_type(cls: Type[Any]) -> bool:
    if not inspect.isclass(cls):
        return False
    return issubclass(cls, tuple) and hasattr(cls, "_fields")


def from_dict(cls, data, override=None):
    if is_namedtuple_type(cls):
        valid_fields = cls._fields
    elif is_dataclass(cls):
        valid_fields = [f.name for f in fields(cls)]
    else:
        raise ValueError(f"Unsupported type: {cls}")

    data_dict = data
    if is_dataclass(data):
        data_dict = dataclasses.asdict(data)
    if override:
        data_dict.update(override)
    filtered_data = {k: v for k, v in data_dict.items() if k in valid_fields}
    return cls(**filtered_data)

@dataclass
class ExecutionConfig:
    execution_id: str 

@dataclass
class ModelConfig:
    sequence_length: int = 1024  # max sequence length
    vocab_size: int = 50257  # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12  # number of layers
    n_head: int = 12  # number of heads
    n_embd: int = 768  # embedding dimension
    batch_size: int = 64
    total_batch_size: int = 524288 # 2**19, ~0.5M, in number of tokens
    

@dataclass
class TrainConfig(ModelConfig, ExecutionConfig):
    max_steps: int = 0
    device: str = 'cpu'
    compile: bool = True
    ddp: bool = False
    ddp_rank: int = 0
    ddp_world_size: int = 1


    rope:bool = False
    ape:bool = True

    infer_interval: int = 1000
    eval_interval: int = 1000
    checkpoint_interval: int = 0
    checkpoint_dir: str = "checkpoints"
    metric_name: str = "loss"
    max_to_keep: int = 5
    mode: str = 'min'

    save_log: bool = True
    logging_dir_prefix: str = "log"

    @cached_property
    def grad_accum_steps(self) -> bool:
        return self.total_batch_size // (self.batch_size * self.sequence_length * self.ddp_world_size)

    @cached_property
    def ddp_master(self) -> bool:
        return self.ddp_rank == 0

    @cached_property
    def infer_enable(self) -> bool:
        return self.infer_interval > 0

    @cached_property
    def eval_enable(self) -> bool:
        return self.eval_interval > 0


@dataclass
class GPTConfig:
    block_size: int = 1024  # max sequence length
    vocab_size: int = 50257  # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12  # number of layers
    n_head: int = 12  # number of heads
    n_embd: int = 768  # embedding dimension


@dataclass
class PicoGPTConfig(TrainConfig):
    preset: str = "gpt2"

    sequence_length: int = 1024
    vocab_size: int = 50304  # 50,000 BPE merges + 256 bytes tokens + 1
    n_layer: int = 12  # number of layers
    n_attn:int = 8
    n_head: int = 12
    n_kv_head: int = 2 # number of key-value heads
    n_embd: int = 768
    bias: bool = False

    max_learning_rate: float = 6e-4
    min_learning_rate: float = max_learning_rate * 0.1
    warmup_steps: int = 715
    max_steps: int = 19073
    weight_decay: float = 0.01

    batch_size: int = 16

    device: str = "cuda"
    ddp: bool = False
    process_rank: int = 0
    local_rank: int = 0
    num_processes: int = 1

defined_presets = {
    "test": {
        "max_steps": 3000, # 1.5B tokens
        "warmup_steps": 500,
        "infer_interval": 0,
        "eval_interval": 300,
        "batch_size": 16
    },
    "debug": {
        "max_steps": 10,
        "infer_interval": 0,
        "eval_interval": 0,
        "checkpoint_interval": 0
    }
}

def get_config(**args) -> PicoGPTConfig:
    execution_id = make_execution_id()

    # TODO - refine device setting for ddp
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"

    override_config = {
        "execution_id": execution_id, 
        "device": device
    }

    preset = args.get("preset")
    if preset in defined_presets:
        override_config.update(defined_presets[preset])

    return from_dict(
        PicoGPTConfig, override_config, args
    )

