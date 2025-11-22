import torch
import os
import glob
from pathlib import Path
from typing import Optional, Dict, Any
import shutil
from src.utils.logging import get_logger

log = get_logger()


class CheckPointManager:
    def __init__(
        self,
        checkpoint_dir: str,
        metric_name: str,
        max_to_keep: int = 5,
        mode: str = 'min'
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.metric_name = metric_name
        self.max_to_keep = max_to_keep
        self.mode = mode
        self.best_metric = float('inf') if mode == 'min' else float('-inf')

    def save_checkpoint(
        self,
        step: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
        metric_value: Optional[float] = None,
        additional_info: Optional[Dict[str, Any]] = None
    ):
        checkpoint = {
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metric_name': self.metric_name,
            'metric_value': metric_value,
        }

        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()

        if additional_info is not None:
            checkpoint.update(additional_info)

        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint-{step}.pth"
        torch.save(checkpoint, checkpoint_path)
        log.info(f"Saved checkpoint to {checkpoint_path}")
        