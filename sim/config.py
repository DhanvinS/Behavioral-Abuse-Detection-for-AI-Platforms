"""Simulation configuration."""
from dataclasses import dataclass, field


@dataclass
class SimConfig:
    n_users: int = 100_000
    days: int = 28
    start_ts: int = 1_777_600_000  # ~2026-05-01 UTC, epoch seconds
    seed: int = 42
    out_dir: str = "data/run"

    # archetype mix (fractions of n_users)
    mix: dict = field(default_factory=lambda: {
        "normal": 0.92,
        "spam_bot": 0.03,
        "account_farmer": 0.02,
        "prompt_sprayer": 0.02,
        "token_thief": 0.01,
    })

    # label noise applied to `label_noisy` (training labels);
    # `label` stays clean for evaluation
    miss_rate: float = 0.03    # abusive users mislabeled as normal (missed enforcement)
    fp_rate: float = 0.003     # normal users mislabeled as abusive (bad reports)

    @property
    def end_ts(self) -> int:
        return self.start_ts + self.days * 86400
