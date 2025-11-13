from typing import NamedTuple

class TrainConfig(NamedTuple):
    model: nn.Module
    optimizer: nn.Module
    lr_scheduler: nn.Module
    max_steps: int
    grad_accum_steps: int
    dataloader: DataLoaderLite
    device_type: str
    ddp: bool
