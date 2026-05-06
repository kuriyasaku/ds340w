from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class RuntimeConfig:
    seed: int = 42
    device: str = "cuda"
    mixed_precision: bool = True
    num_workers: int = 8
    pin_memory: bool = True


@dataclass
class DataConfig:
    tile_size: int = 256
    stride: int = 256
    tissue_threshold: float = 0.55
    max_tiles_per_slide: int = 12000
    bag_size: int = 2048
    label_column: str = "label"
    slide_column: str = "slide_id"
    split_column: str = "split"
    coordinate_columns: List[str] = field(default_factory=lambda: ["x", "y"])


@dataclass
class FeatureConfig:
    model_name: str = "owkin/phikon"
    feature_dim: int = 768
    batch_size: int = 96
    normalize: bool = True
    pooling: str = "cls"


@dataclass
class MILConfig:
    input_dim: int = 768
    hidden_dim: int = 512
    attention_dim: int = 256
    dropout: float = 0.25
    num_classes: int = 2
    top_k_ratio: float = 0.05


@dataclass
class TrainConfig:
    epochs: int = 30
    batch_size: int = 1
    lr: float = 1e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 2
    gradient_clip: float = 2.0
    early_stop_patience: int = 6
    save_every: int = 1


@dataclass
class AuditConfig:
    perturbations: List[str] = field(default_factory=lambda: ["blur", "stain_shift", "artifacts", "color_removal", "brightness_shift"])
    blur_kernel: int = 15
    brightness_gamma: float = 0.85
    stain_reduction: float = 0.15
    artifact_ratio: float = 0.05
    explanation_threshold: float = 0.5


@dataclass
class AppConfig:
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    mil: MILConfig = field(default_factory=MILConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    audit: AuditConfig = field(default_factory=AuditConfig)


CONFIG = AppConfig()
