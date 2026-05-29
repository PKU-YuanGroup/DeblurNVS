__all__ = ["DeblurNVSPipeline"]


def __getattr__(name):
    if name == "DeblurNVSPipeline":
        from .pipeline import DeblurNVSPipeline

        return DeblurNVSPipeline
    raise AttributeError(name)
