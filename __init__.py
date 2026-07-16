"""LNR_HELPER — ComfyUI custom node package."""

import os, sys, importlib

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

for fn in sorted(os.listdir(ROOT)):
    if not fn.endswith(".py") or fn.startswith("__"):
        continue
    mod = fn[:-3]
    try:
        m = importlib.import_module(mod)
        if hasattr(m, "NODE_CLASS_MAPPINGS"):
            NODE_CLASS_MAPPINGS.update(m.NODE_CLASS_MAPPINGS)
        if hasattr(m, "NODE_DISPLAY_NAME_MAPPINGS"):
            NODE_DISPLAY_NAME_MAPPINGS.update(m.NODE_DISPLAY_NAME_MAPPINGS)
    except Exception:
        pass

WEB_DIRECTORY = os.path.join(ROOT, "js")
