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

        # Update best checkpoint if applicable
        if metric_value is not None:
            is_best = False
            if self.mode == "min" and metric_value < self.best_metric:
                is_best = True
                self.best_metric = metric_value
            elif self.mode == "max" and metric_value > self.best_metric:
                is_best = True
                self.best_metric = metric_value
                
            if is_best:
                best_path = self.checkpoint_dir / "best_checkpoint.pt"
                shutil.copy2(checkpoint_path, best_path)
                log.info(f"New best checkpoint! {self.metric_name}: {metric_value:.4f}")
        
        # Clean up old checkpoints (keep only N most recent)
        self._cleanup_old_checkpoints()

    def _cleanup_old_checkpoints(self):
        # List all checkpoints in the directory
        checkpoint_files = glob.glob(str(self.checkpoint_dir / "checkpoint-*.pth"))
        checkpoint_files.sort(key=lambda x: int(x.split("-")[-1].split(".")[0]))
        
        # Keep only the N most recent checkpoints
        if len(checkpoint_files) > self.max_to_keep:
            for old_checkpoint in checkpoint_files[:-self.max_to_keep]:
                try:
                    os.remove(old_checkpoint)
                    log.info(f"Removed old checkpoint: {old_checkpoint}")
                except Exception as e:
                    log.error(f"Failed to remove old checkpoint {old_checkpoint}: {e}")
    
    def load_checkpoint(
            self,
            model: torch.nn.Module,
            optimizer: Optional[torch.optim.Optimizer] = None,
            checkpoint_path: Optional[str] = None,
            load_best: bool = False
        ) -> Dict[str, Any]:
        """
        Load a checkpoint and restore model/optimizer/scheduler states.
        
        Args:
            model: PyTorch model to load state into
            optimizer: PyTorch optimizer to load state into (optional)
            scheduler: Learning rate scheduler to load state into (optional)
            checkpoint_path: Specific checkpoint path to load (optional)
            load_best: If True, load best checkpoint (optional)
            
        Returns:
            Dictionary containing checkpoint information (epoch, metric, etc.)
        """
        if load_best:
            checkpoint_path = self.checkpoint_dir / "best_checkpoint.pt"
        elif checkpoint_path is None:
            # Load most recent checkpoint
            checkpoint_files = glob.glob(str(self.checkpoint_dir / "checkpoint_epoch_*.pt"))
            if not checkpoint_files:
                raise FileNotFoundError(f"No checkpoints found in {self.checkpoint_dir}")
            checkpoint_path = max(checkpoint_files, key=os.path.getmtime)
        
        log.info(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        log.info(f"Loaded checkpoint from step {checkpoint['step']}")
        if checkpoint.get('metric_value') is not None:
            log.info(f"{checkpoint['metric_name']}: {checkpoint['metric_value']:.4f}")
        
        return checkpoint