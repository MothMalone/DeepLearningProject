"""Configuration objects for the Vietnamese GPT-2 math experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


SAFE_EOS_ID = 50256
PROMPT_TEMPLATE = "Câu hỏi: {q}\nGiải:\n"


@dataclass(frozen=True)
class PathConfig:
    """Kaggle-compatible paths used by the final notebook."""

    data_dir: Path
    model_name: str
    output_dir: Path = Path("/kaggle/working/gpt2_math_ckpt")
    valid_output_path: Path = Path("/kaggle/working/valid_output.json")
    valid_report_path: Path = Path("/kaggle/working/valid_report.json")
    baseline_output_path: Path = Path("/kaggle/working/baseline_valid_output.json")
    baseline_report_path: Path = Path("/kaggle/working/baseline_valid_report.json")
    test_output_path: Path = Path("/kaggle/working/test_predictions.json")
    experiment_report_path: Path = Path("/kaggle/working/experiment_report.json")
    error_analysis_path: Path = Path("/kaggle/working/error_analysis.json")
    hpo_report_path: Path = Path("/kaggle/working/hpo_report.json")

    @property
    def train_file(self) -> Path:
        return self.data_dir / "train.json"

    @property
    def valid_file(self) -> Path:
        return self.data_dir / "valid.json"

    @property
    def test_file(self) -> Path:
        return self.data_dir / "test.json"


@dataclass(frozen=True)
class TrainingConfig:
    """Final rewind3-style full fine-tuning configuration."""

    epochs: int = 8
    lr: float = 8e-3
    warmup_ratio: float = 0.15
    max_length: int = 256
    max_response_tokens: int = 256
    per_device_batch_size: int = 8
    grad_accum: int = 4
    weight_decay: float = 0.01
    seed: int = 42
    computation_weight: float = 2.5
    anchor_weight: float = 6.0
    eos_weight: float = 8.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DecodingConfig:
    """Self-consistency decoding configuration that performed best locally."""

    max_new_tokens: int = 80
    use_beam_search: bool = False
    num_beams: int = 5
    num_samples: int = 15
    temperature: float = 0.4
    top_k: int = 40
    top_p: float = 0.90

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


DECODING_GRID: list[tuple[str, dict[str, Any]]] = [
    ("beam3", {"use_beam": True, "num_beams": 3, "sc": False}),
    ("beam5", {"use_beam": True, "num_beams": 5, "sc": False}),
    ("beam7", {"use_beam": True, "num_beams": 7, "sc": False}),
    ("sc5_t04", {"use_beam": False, "sc": True, "n": 5, "temp": 0.4, "top_k": 40, "top_p": 0.90}),
    ("sc9_t04", {"use_beam": False, "sc": True, "n": 9, "temp": 0.4, "top_k": 40, "top_p": 0.90}),
    ("sc9_t03", {"use_beam": False, "sc": True, "n": 9, "temp": 0.3, "top_k": 30, "top_p": 0.85}),
    ("sc9_t05", {"use_beam": False, "sc": True, "n": 9, "temp": 0.5, "top_k": 50, "top_p": 0.95}),
    ("sc15_t04", {"use_beam": False, "sc": True, "n": 15, "temp": 0.4, "top_k": 40, "top_p": 0.90}),
]
