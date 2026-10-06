"""v3.2 dev: agent-written appearance operators (sandboxed).

The agent may attach, per class, Python source defining

    def augment(pasted, target, mask, rng):
        # pasted, target: HxWx3 uint8 RGB (after / before the paste), mask: HxW bool
        # (the inserted defect), rng: numpy Generator.  Return an HxWx3 image.

It runs after the Copy-Paste and before the boundary feather.  Only pixels inside
`mask` may change (enforced: everything else is reset to `pasted`), so labels and
masks stay exact; the QC gate (inside-mask change) is re-checked on the result.

Sandbox: the source is parsed and rejected if it imports, touches dunder or
private attributes, or names file/OS/introspection functions; it runs with a
small builtins whitelist and only `np` and `cv2` in scope, under a per-call time
limit.  This guards against accidents by a local model, not against an attacker.
"""

from __future__ import annotations

import ast
import time
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

CALL_SECONDS = 2.0
_BANNED_NAMES = {
    "open", "exec", "eval", "compile", "globals", "locals", "vars", "dir", "getattr", "setattr",
    "delattr", "input", "breakpoint", "help", "memoryview", "type", "object", "super", "__import__",
    "exit", "quit", "classmethod", "staticmethod", "property",
}
_BANNED_ATTRS = {
    "load", "save", "savez", "savez_compressed", "savetxt", "loadtxt", "genfromtxt", "fromfile", "tofile",
    "memmap", "imwrite", "imread", "imreadmulti", "VideoCapture", "VideoWriter", "system", "ctypes",
    "lib", "os", "sys", "f2py", "testing", "distutils", "DataSource", "dump", "dumps", "Formatter",
}
_BUILTINS = {n: __builtins__[n] if isinstance(__builtins__, dict) else getattr(__builtins__, n)
             for n in ("abs", "min", "max", "range", "len", "float", "int", "round", "sum", "zip", "enumerate",
                       "list", "tuple", "dict", "bool", "isinstance", "ValueError", "any", "all", "sorted",
                       "reversed", "map", "filter", "pow", "divmod", "set", "slice", "print")}
_BUILTINS["print"] = lambda *a, **k: None


class OpError(ValueError):
    pass


def check_source(src: str) -> None:
    try:
        tree = ast.parse(src)
    except SyntaxError as exc:
        raise OpError(f"syntax error: {exc}") from None
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise OpError("imports are not allowed (np and cv2 are already available)")
        if isinstance(node, (ast.Global, ast.Nonlocal, ast.With, ast.AsyncWith, ast.AsyncFunctionDef,
                             ast.ClassDef, ast.Lambda, ast.Yield, ast.YieldFrom, ast.Await)):
            raise OpError(f"{type(node).__name__} is not allowed")
        if isinstance(node, ast.Name) and (node.id in _BANNED_NAMES or node.id.startswith("_")):
            raise OpError(f"name {node.id!r} is not allowed")
        if isinstance(node, ast.Attribute) and (node.attr.startswith("_") or node.attr in _BANNED_ATTRS):
            raise OpError(f"attribute {node.attr!r} is not allowed")
        if isinstance(node, ast.While):
            raise OpError("while loops are not allowed (use numpy operations)")
    names = [n.name for n in tree.body if isinstance(n, ast.FunctionDef)]
    if "augment" not in names:
        raise OpError("define a function named augment(pasted, target, mask, rng)")


def compile_op(src: str) -> Callable:
    check_source(src)
    scope: dict = {"__builtins__": _BUILTINS, "np": np, "cv2": cv2}
    exec(compile(src, "<agent_op>", "exec"), scope)   # noqa: S102 -- source checked above
    return scope["augment"]


def apply_op(fn: Callable, pasted: np.ndarray, target: np.ndarray, mask: np.ndarray,
             rng: np.random.Generator) -> np.ndarray:
    t = time.perf_counter()
    try:
        out = fn(pasted.copy(), target.copy(), mask.copy(), rng)
    except OpError:
        raise
    except Exception as exc:   # agent code: any failure is a rejected candidate
        raise OpError(f"augment raised {type(exc).__name__}: {exc}") from None
    if time.perf_counter() - t > CALL_SECONDS:
        raise OpError(f"augment took more than {CALL_SECONDS}s per image")
    out = np.asarray(out)
    if out.shape != pasted.shape:
        raise OpError(f"augment must return shape {pasted.shape}, got {out.shape}")
    out = np.clip(np.rint(out.astype(np.float64)), 0, 255).astype(np.uint8) if out.dtype != np.uint8 else out
    result = pasted.copy()
    result[mask] = out[mask]          # only the inserted defect may change
    return result


def inside_change_fraction(image: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    delta = np.abs(image.astype(np.int16) - target.astype(np.int16)).max(axis=2)
    return float((delta[mask] >= 3).mean())


def mean_inside_contrast(image: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    return float(np.abs(image.astype(np.int16) - target.astype(np.int16)).max(axis=2)[mask].mean())


def probe(src: str, sources: list[Path], masks_dir: Path, labels_dir: Path, class_id: int,
          n: int = 8, seed: int = 0, gate: float = 0.5) -> dict:
    """Run the op on n real pastes of this class; returns QC pass rate and contrast before/after."""
    from defectaug.patch_copy import copy_real_defect, read_binary_mask, read_rgb, read_yolo_boxes
    fn = compile_op(src)
    rng = np.random.default_rng(seed)
    passed, before, after, errors = 0, [], [], 0
    for i in range(n):
        s, t = sources[i % len(sources)], sources[(i * 7 + 3) % len(sources)]
        try:
            res = copy_real_defect(source_image=read_rgb(s), source_mask=read_binary_mask(masks_dir / f"{s.stem}.png"),
                                   target_image=read_rgb(t), target_boxes=read_yolo_boxes(labels_dir / f"{t.stem}.txt"),
                                   class_id=class_id, rng=rng, scale=0.6, max_overlap_fraction=0.3,
                                   placement_attempts=256)
        except (ValueError, RuntimeError):
            errors += 1
            continue
        target = read_rgb(t)
        img = apply_op(fn, res.image, target, res.inserted_mask, rng)
        before.append(mean_inside_contrast(res.image, target, res.inserted_mask))
        after.append(mean_inside_contrast(img, target, res.inserted_mask))
        passed += inside_change_fraction(img, target, res.inserted_mask) >= gate
    k = len(before)
    return {"samples": k, "qc_pass": passed, "mean_contrast_before": round(float(np.mean(before)), 1) if k else None,
            "mean_contrast_after": round(float(np.mean(after)), 1) if k else None}
