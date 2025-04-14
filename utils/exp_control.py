# experiment control utilities

import os
import json
import time
import torch
import logging
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Union
from torch.utils.tensorboard import SummaryWriter  # Add this import

class ExperimentController:
    """Controls experiment execution, logging, checkpointing, and visualization."""
    
    def __init__(
        self,
        experiment_name: str,
        base_dir: str = "experiments",
        config: Optional[Dict[str, Any]] = None,
        resume_from: Optional[str] = None
    ):
        """
        Initialize experiment controller.
        
        Args:
            experiment_name: Name of the experiment
            base_dir: Base directory for all experiments
            config: Configuration dictionary to save with experiment
            resume_from: Path to resume experiment from
        """
        self.experiment_name = experiment_name
        self.base_dir = base_dir
        
        # Setup directories
        if resume_from:
            self.exp_dir = Path(resume_from)
            self.config = self._load_config()
            
            self.logs_dir = self.exp_dir / "logs"
            self.checkpoints_dir = self.exp_dir / "checkpoints"
            self.images_dir = self.exp_dir / "images"
            
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.exp_dir = Path(base_dir) / f"{experiment_name}_{timestamp}"
            self.exp_dir.mkdir(parents=True, exist_ok=True)
            
            # Create subdirectories
            self.checkpoints_dir = self.exp_dir / "checkpoints"
            self.checkpoints_dir.mkdir(exist_ok=True)
            
            self.images_dir = self.exp_dir / "images"
            self.images_dir.mkdir(exist_ok=True)
            
            self.logs_dir = self.exp_dir / "logs"
            self.logs_dir.mkdir(exist_ok=True)
            
            # Save config if provided
            self.config = config or {}
            self._save_config()
        
        # Initialize metrics
        self.metrics = {'train': {}, 'val': {}}
        self.best_metric_value = float('inf')
        self.start_time = time.time()
        
        # Setup logging
        self._setup_logger()
        
        # Initialize TensorBoard writer
        self.tb_log_dir = self.exp_dir / "tensorboard"
        self.tb_log_dir.mkdir(exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.tb_log_dir))
        
        # Log hyperparameters to TensorBoard if config exists
        if self.config:
            # Extract only serializable parameters for TensorBoard
            hparams = {k: v for k, v in self.config.items() 
                      if isinstance(v, (int, float, str, bool))}
            self.writer.add_hparams(hparams, {})
        
        self.logger.info(f"Experiment initialized at {self.exp_dir}")
        self.logger.info(f"TensorBoard logs available at {self.tb_log_dir}")
        if config:
            self.logger.info(f"Configuration: {json.dumps(config, indent=2)}")
    
    def _setup_logger(self):
        """Set up the experiment logger."""
        self.logger = logging.getLogger(self.experiment_name)
        self.logger.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # File handler
        file_handler = logging.FileHandler(self.logs_dir / "experiment.log")
        file_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        
        # Add handlers
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
    
    def _save_config(self):
        """Save experiment configuration."""
        with open(self.exp_dir / "config.json", 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def _load_config(self):
        """Load experiment configuration."""
        config_path = self.exp_dir / "config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                return json.load(f)
        return {}
    
    def log_metrics(self, metrics: Dict[str, float], step: int, phase: str = 'train'):
        """
        Log metrics for the current step.
        
        Args:
            metrics: Dictionary of metric names and values
            step: Current step number
            phase: 'train' or 'val'
        """
        for name, value in metrics.items():
            if name not in self.metrics[phase]:
                self.metrics[phase][name] = []
            
            # Store as (step, value) pairs
            self.metrics[phase][name].append((step, value))
            
            # Log to TensorBoard
            self.writer.add_scalar(f"{phase}/{name}", value, step)
        
        # Log to file
        metrics_str = ', '.join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.logger.info(f"[{phase.upper()}] Step {step}: {metrics_str}")
        
        # Save metrics to disk
        self._save_metrics()
    
    def _save_metrics(self):
        """Save metrics to disk."""
        metrics_data = {}
        for phase in self.metrics:
            metrics_data[phase] = {
                metric: [(step, float(value)) for step, value in values]
                for metric, values in self.metrics[phase].items()
            }
        
        with open(self.logs_dir / "metrics.json", 'w') as f:
            json.dump(metrics_data, f, indent=2)
    
    def save_model(self, model, optimizer=None, scheduler=None, step=None, is_best=False):
        """
        Save model checkpoint.
        
        Args:
            model: PyTorch model to save
            optimizer: Optional optimizer state to save
            scheduler: Optional scheduler state to save
            step: Current training step
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "step": step,
            "timestamp": datetime.now().isoformat()
        }
        
        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()
        
        if scheduler is not None:
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()
        
        # Save checkpoint
        filename = f"checkpoint_{step}.pt" if step else "latest.pt"
        checkpoint_path = self.checkpoints_dir / filename
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Model saved to {checkpoint_path}")
        
        # Save best model if specified
        if is_best:
            best_path = self.checkpoints_dir / "best.pt"
            torch.save(checkpoint, best_path)
            self.logger.info(f"Best model saved to {best_path}")
    
    def load_model(self, model, checkpoint_path=None, optimizer=None, scheduler=None):
        """
        Load model from checkpoint.
        
        Args:
            model: PyTorch model to load weights into
            checkpoint_path: Path to checkpoint file (or 'best' or 'latest')
            optimizer: Optional optimizer to load state into
            scheduler: Optional scheduler to load state into
            
        Returns:
            step: The training step from the checkpoint
        """
        if checkpoint_path is None or checkpoint_path == 'latest':
            checkpoint_path = self.checkpoints_dir / "latest.pt"
        elif checkpoint_path == 'best':
            checkpoint_path = self.checkpoints_dir / "best.pt"
        
        if not os.path.exists(checkpoint_path):
            self.logger.warning(f"Checkpoint {checkpoint_path} not found!")
            return None
        
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        if scheduler is not None and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        step = checkpoint.get("step", None)
        self.logger.info(f"Model loaded from {checkpoint_path} (step {step})")
        return step
    
    def save_image(self, img_tensor, filename, normalize=True, to_tensorboard=True, tag=None):
        """
        Save image tensor to disk and optionally to TensorBoard.
        
        Args:
            img_tensor: Image tensor to save
            filename: Filename for the image
            normalize: Whether to normalize the image
            to_tensorboard: Whether to log the image to TensorBoard
            tag: Tag for TensorBoard (defaults to filename without extension)
            
        Returns:
            str: Path to saved image file
        """
        import torchvision
        path = self.images_dir / filename
        torchvision.utils.save_image(img_tensor, path, normalize=normalize)
        
        # Log to TensorBoard if requested
        if to_tensorboard:
            if tag is None:
                tag = Path(filename).stem  # Use filename without extension
                
            # Get global step if it's in the filename (e.g., "result_100.png" -> step 100)
            step = None
            try:
                import re
                match = re.search(r'_(\d+)', tag)
                if match:
                    step = int(match.group(1))
            except:
                step = None
                
            # Add image to TensorBoard
            self.writer.add_image(tag, img_tensor, global_step=step, dataformats='NCHW')
            
        return str(path)
    
    def plot_metrics(self, metrics_list=None, phases=None, steps=None, smooth=0):
        """
        Plot training metrics.
        
        Args:
            metrics_list: List of metric names to plot (None = all)
            phases: List of phases to include ('train', 'val')
            steps: Range of steps to plot (None = all)
            smooth: Smoothing factor (0 = no smoothing)
        """
        phases = phases or ['train', 'val']
        metrics_list = metrics_list or []
        
        # Collect all metric names if not specified
        if not metrics_list:
            for phase in phases:
                metrics_list.extend(self.metrics[phase].keys())
            metrics_list = list(set(metrics_list))  # Remove duplicates
        
        # Create plots
        for metric in metrics_list:
            plt.figure(figsize=(10, 5))
            for phase in phases:
                if metric not in self.metrics[phase]:
                    continue
                
                # Extract steps and values
                steps_values = self.metrics[phase][metric]
                if not steps_values:
                    continue
                    
                x, y = zip(*steps_values)
                
                # Apply smoothing if needed
                if smooth > 0 and len(y) > smooth:
                    kernel = np.ones(smooth) / smooth
                    y_smooth = np.convolve(y, kernel, mode='valid')
                    x_smooth = x[smooth-1:]
                    plt.plot(x_smooth, y_smooth, label=f"{phase} (smoothed)")
                
                # Plot raw data
                plt.plot(x, y, label=f"{phase}", alpha=0.5 if smooth > 0 else 1.0)
            
            plt.xlabel("Steps")
            plt.ylabel(metric)
            plt.title(f"{metric} over training")
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Save figure
            save_path = self.logs_dir / f"{metric}_plot.png"
            plt.savefig(save_path)
            plt.close()
            
        self.logger.info(f"Metric plots saved to {self.logs_dir}")
    
    def update_best_model(self, model, optimizer, metric_value, step, lower_is_better=True):
        """
        Update the best model if the metric is improved.
        
        Args:
            model: PyTorch model
            optimizer: Optimizer
            metric_value: Current metric value
            step: Current step
            lower_is_better: Whether lower metric is better
            
        Returns:
            bool: True if this is a new best model
        """
        is_best = False
        if (lower_is_better and metric_value < self.best_metric_value) or \
           (not lower_is_better and metric_value > self.best_metric_value):
            self.best_metric_value = metric_value
            is_best = True
            self.save_model(model, optimizer, step=step, is_best=True)
            self.logger.info(f"New best model with {metric_value:.4f} at step {step}")
        
        return is_best
    
    def finish(self):
        """
        Finalize the experiment (save final metrics plots, etc.)
        """
        duration = time.time() - self.start_time
        self.logger.info(f"Experiment finished. Duration: {duration:.2f} seconds")
        
        # Plot final metrics
        self.plot_metrics()
    
    def log_model_graph(self, model, input_shape=(1, 3, 224, 224)):
        """
        Log model architecture to TensorBoard.
        
        Args:
            model: PyTorch model
            input_shape: Input shape for model visualization (batch_size, channels, height, width)
        """
        try:
            device = next(model.parameters()).device
            dummy_input = torch.zeros(input_shape, device=device)
            self.writer.add_graph(model, dummy_input)
            self.logger.info(f"Model graph added to TensorBoard")
        except Exception as e:
            self.logger.warning(f"Failed to add model graph to TensorBoard: {e}")

