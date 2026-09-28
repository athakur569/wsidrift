"""CycleGAN stain transfer stub for wsidrift E4/E7/E9 conditions."""
from .generator import ResNetGenerator
from .trainer import CycleGANTrainer

__all__ = ["ResNetGenerator", "CycleGANTrainer"]
