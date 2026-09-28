"""Ablation switches for the Jev study, settable from the command line.

Every default reproduces the unmodified NAAMSE behavior, so a run with no
flags is the baseline arm.
"""
import argparse
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

from langchain_core.runnables import RunnableConfig

ACTION_SELECTORS = ("static_thresholds", "uniform", "jev")
MUTATION_SELECTORS = ("uniform", "jev")
JUDGES = ("gemini", "meta", "jev")
REFEREES = ("meta", "gemini", "none")
OBJECTIVES = ("score", "coverage", "combined")


@dataclass(frozen=True)
class AblationConfig:
    """One arm of the ablation grid."""
    action_selector: str = "static_thresholds"
    mutation_selector: str = "uniform"
    fitness_judge: str = "gemini"
    referee_judge: str = "none"
    objective: str = "score"
    coverage_weight: float = 0.5
    jev_temperature: float = 1.0

    def __post_init__(self):
        _check("action_selector", self.action_selector, ACTION_SELECTORS)
        _check("mutation_selector", self.mutation_selector, MUTATION_SELECTORS)
        _check("fitness_judge", self.fitness_judge, JUDGES)
        _check("referee_judge", self.referee_judge, REFEREES)
        _check("objective", self.objective, OBJECTIVES)
        if not 0.0 <= self.coverage_weight <= 1.0:
            raise ValueError(f"coverage_weight must be in [0, 1], got {self.coverage_weight}")
        if self.jev_temperature < 0:
            raise ValueError(f"jev_temperature must be >= 0, got {self.jev_temperature}")

    @property
    def arm_name(self) -> str:
        """Short, filesystem-safe name for this arm."""
        return (f"act-{self.action_selector}_mut-{self.mutation_selector}"
                f"_fit-{self.fitness_judge}_obj-{self.objective}")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _check(name: str, value: str, allowed: tuple) -> None:
    if value not in allowed:
        raise ValueError(f"{name} must be one of {allowed}, got {value!r}")


def get_ablation(config: Optional[RunnableConfig]) -> AblationConfig:
    """Read the ablation arm from a LangGraph config; defaults to the baseline."""
    if not config:
        return AblationConfig()
    return config.get("configurable", {}).get("ablation") or AblationConfig()


def add_ablation_args(parser: argparse.ArgumentParser) -> None:
    """Register the ablation flags on an argparse parser."""
    group = parser.add_argument_group("ablation")
    group.add_argument("--action-selector", choices=ACTION_SELECTORS, default="static_thresholds",
                       help="RQ1: how EXPLORE/SIMILAR/MUTATE is chosen")
    group.add_argument("--mutation-selector", choices=MUTATION_SELECTORS, default="uniform",
                       help="RQ3: how the mutation operator is chosen")
    group.add_argument("--fitness-judge", choices=JUDGES, default="gemini",
                       help="RQ2: judge that produces the fuzzer's fitness signal")
    group.add_argument("--referee-judge", choices=REFEREES, default="none",
                       help="Fixed judge that re-scores every arm's outputs for evaluation "
                            "(recorded in config.json; the runner does not re-score yet)")
    group.add_argument("--objective", choices=OBJECTIVES, default="score",
                       help="Fitness objective: judge score, category coverage, or both")
    group.add_argument("--coverage-weight", type=float, default=0.5,
                       help="Weight of coverage in the combined objective")
    group.add_argument("--jev-temperature", type=float, default=1.0,
                       help="Sampling temperature over Jev probabilities (0 = argmax)")


def ablation_from_args(args: argparse.Namespace) -> AblationConfig:
    return AblationConfig(
        action_selector=args.action_selector,
        mutation_selector=args.mutation_selector,
        fitness_judge=args.fitness_judge,
        referee_judge=args.referee_judge,
        objective=args.objective,
        coverage_weight=args.coverage_weight,
        jev_temperature=args.jev_temperature,
    )
