try:
    from .rae_da3 import RAE_DA3
except (ImportError, ModuleNotFoundError) as _e:
    import warnings
    warnings.warn(f"RAE_DA3 unavailable ({_e}). DA3 backbone will not work.")
    RAE_DA3 = None

try:
    from .rae_vggt import RAE_VGGT
except (ImportError, AttributeError):
    RAE_VGGT = None
