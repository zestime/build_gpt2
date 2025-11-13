from functools import cached_property
from pydantic import BaseModel
from utils import make_execution_id
from typing import Any, Type
from dataclasses import dataclass, is_dataclass, fields
import dataclasses
import inspect
import torch


def is_namedtuple_type(cls: Type[Any]) -> bool:
    if not inspect.isclass(cls):
        return False
    return issubclass(cls, tuple) and hasattr(cls, "_fields")


def from_dict(cls, data):
    print(data)
    if is_namedtuple_type(cls):
        valid_fields = cls._fields
    elif is_dataclass(cls):
        valid_fields = [f.name for f in fields(cls)]
    else:
        raise ValueError(f"Unsupported type: {cls}")

    data_dict = data
    if is_dataclass(data):
        data_dict = dataclasses.asdict(data)
    filtered_data = {k: v for k, v in data_dict.items() if k in valid_fields}
    return cls(**filtered_data)


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
class TrainConfig(ModelConfig):
    max_steps: int = 0
    device: str = 'cpu'
    compile: bool = True
    ddp: bool = False
    ddp_rank: int = 0
    ddp_world_size: int = 1

    @cached_property
    def grad_accum_steps(self) -> bool:
        return self.total_batch_size // (self.batch_size * self.sequence_length * self.ddp_world_size)

    @cached_property
    def ddp_master(self) -> bool:
        return self.ddp_rank == 0


@dataclass
class GPTConfig:
    block_size: int = 1024  # max sequence length
    vocab_size: int = 50257  # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12  # number of layers
    n_head: int = 12  # number of heads
    n_embd: int = 768  # embedding dimension


@dataclass
class PicoGPTConfig:
    execution_id: str
    preset: str = "gpt2"

    sequence_length: int = 1024
    vocab_size: int = 50304  # 50,000 BPE merges + 256 bytes tokens + 1
    n_layer: int = 12  # number of layers
    n_head: int = 12
    n_embd: int = 768
    bias: bool = False

    max_learning_rate: float = 6e-4
    min_learning_rate: float = max_learning_rate * 0.1
    warmup_steps: int = 715
    max_steps: int = 19073
    weight_decay: float = 0.01

    batch_size: int = 32

    device: str = "cuda"
    ddp: bool = False
    process_rank: int = 0
    local_rank: int = 0
    num_processes: int = 1

    save_log: bool = True
    logging_dir_prefix: str = "log"
    save_checkpoint: int = 1000
    save_optimizer: bool = True


class OptimizerConfig(BaseModel):
    max_learning_rate: float = 6e-4
    min_learning_rate: float = max_learning_rate * 0.1
    warmup_steps: int = 715
    max_steps: int = 19073
    weight_decay: float = 0.01
    save_checkpoint: int = 1000
    save_optimizer: bool = True


def get_config(**args) -> PicoGPTConfig:
    execution_id = make_execution_id()

    # TODO - refine device setting for ddp
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"

    default_config = {
        "execution_id": execution_id, "device": device
    }

    data = args | default_config

    return from_dict(
        PicoGPTConfig, data
    )
