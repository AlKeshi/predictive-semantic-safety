from importlib import import_module

__all__ = ["Predictor", "Forecast", "SafetyFilter", "run_demo", "run_benchmark", "load_case"]
_API = {
    "Predictor": ("pss.predictors.code_as_world", "create_predictor"),
    "Forecast": ("pss.control.forecast", "Forecast"),
    "SafetyFilter": ("pss.control.code_world", "CodeWorldController"),
    "run_demo": ("demos.api", "run_demo"),
    "run_benchmark": ("benchmarks.run", "run"),
    "load_case": ("datagen", "load_case"),
}


def __getattr__(name):
    if name not in _API:
        raise AttributeError(name)
    module, attribute = _API[name]
    return getattr(import_module(module), attribute)
