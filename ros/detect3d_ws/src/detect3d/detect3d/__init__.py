# detect3d/__init__.py
import sys
import importlib

# Expose detect3d.heads → heads, detect3d.BEV → BEV, detect3d.pkgs → pkgs
for alias in ['heads', 'BEV', 'pkgs']:
    try:
        sys.modules[alias] = importlib.import_module(f"{__name__}.{alias}")
    except Exception as e:
        print(f"[WARN] Could not create alias for {alias}: {e}")
