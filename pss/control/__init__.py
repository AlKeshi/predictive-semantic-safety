from .flow import braking_flow
from .forecast import Forecast

__all__ = ["CodeWorldController", "Forecast", "braking_flow", "RetreatConfig"]


def __getattr__(name):
    if name == "RetreatConfig":
        from .retreat import RetreatConfig

        return RetreatConfig
    if name == "CodeWorldController":
        from .code_world import CodeWorldController

        return CodeWorldController
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
