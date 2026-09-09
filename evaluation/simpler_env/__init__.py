from .adaptive_ensemble import AdaptiveEnsembler

__all__ = ["AdaptiveEnsembler", "VLAInference"]


def __getattr__(name):
    """Avoid importing the full SimplerEnv dependency stack for LIBERO eval."""
    if name == "VLAInference":
        from .vla_policy import VLAInference

        return VLAInference
    raise AttributeError(name)
