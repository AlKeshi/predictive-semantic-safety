from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True, kw_only=True)
class EpisodeConfig:
    output: Path
    duration: float
    seed: int
    speed: float
    model: str | Path | None = None
    render: bool = False
    device: str = "cuda"
    navigation_profile: Literal["benchmark", "demo-ground-truth"] = "benchmark"
    demo_margin_m: float = 0.0
    scene_xml: str | None = None


@dataclass(frozen=True, kw_only=True)
class CeilingConfig(EpisodeConfig):
    method: Literal["cw", "omnivla"]


@dataclass(frozen=True, kw_only=True)
class RouteConfig(EpisodeConfig):
    scene: str
    method: Literal["code_world", "nominal"]
    warmup: float
    horizon: float = 2.5
    update_period: float = 1.0
