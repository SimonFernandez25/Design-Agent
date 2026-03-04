"""
dot_path_editor.py - Deterministic Dot-Path Design Editor

Applies a flat dict of dot-separated key paths to a nested JSON design dict.

Rules (from architecture spec):
- Traverse nested JSON using dot-path notation.
- Reject invalid paths with a clear KeyError.
- Reject type mismatches (e.g. writing a float to an int key) with TypeError.
- Log each change to a returned change list.
- NEVER calls an LLM.
- NEVER regenerates the design.
- Returns a deep copy -- caller's dict is never mutated.

Example edit dict:
    {
        "objects.triangular_pyramid.dimensions.base_width_um": 12,
        "assembly.grid.x": 8,
        "objects.triangular_pyramid.construction.layers": 30
    }
"""

import copy
from typing import Any, Dict, List, Tuple


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _traverse(root: dict, parts: List[str]) -> Tuple[Any, str]:
    """
    Walk *root* following *parts[:-1]* and return
    (parent_container, last_key) so the caller can assign into it.

    Raises
    ------
    KeyError
        If any intermediate key does not exist.
    TypeError
        If an intermediate node is not a dict (cannot traverse into it).
    """
    node: Any = root
    for part in parts[:-1]:
        if not isinstance(node, dict):
            raise TypeError(
                f"Cannot traverse into non-dict node at key '{part}'. "
                f"Node type: {type(node).__name__}"
            )
        if part not in node:
            raise KeyError(
                f"Key '{part}' not found. "
                f"Available keys: {list(node.keys())}"
            )
        node = node[part]

    # Final container must be a dict
    if not isinstance(node, dict):
        raise TypeError(
            f"Parent of target key is not a dict (type: {type(node).__name__})"
        )

    last_key = parts[-1]
    if last_key not in node:
        raise KeyError(
            f"Target key '{last_key}' not found. "
            f"Available keys: {list(node.keys())}"
        )

    return node, last_key


def _check_type_compatibility(existing: Any, new_value: Any, path: str) -> None:
    """
    Reject type mismatches that would silently corrupt the design.

    Allowed coercions:
    - int -> float and float -> int are both permitted (numeric family).
    - str  -> str  only.
    - bool -> bool only (bool is subclass of int in Python, handle first).
    - list -> list only.
    - dict -> dict only.
    """
    if existing is None:
        return  # None sentinel; allow any type

    if isinstance(existing, bool) or isinstance(new_value, bool):
        if type(existing) is not type(new_value):
            raise TypeError(
                f"Type mismatch at '{path}': "
                f"existing type is {type(existing).__name__!r}, "
                f"new value type is {type(new_value).__name__!r}. "
                f"Bool fields must remain bool."
            )
        return

    if isinstance(existing, (int, float)) and isinstance(new_value, (int, float)):
        return  # numeric family; always ok

    if type(existing) is not type(new_value):
        raise TypeError(
            f"Type mismatch at '{path}': "
            f"existing type is {type(existing).__name__!r}, "
            f"new value type is {type(new_value).__name__!r}. "
            f"To change the type intentionally use apply_dot_path_edits with strict=False."
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def apply_dot_path_edits(
    design_json: Dict[str, Any],
    edits: Dict[str, Any],
    strict: bool = True,
) -> Tuple[Dict[str, Any], List[str]]:
    """
    Apply a flat dict of dot-path edits to *design_json*.

    Parameters
    ----------
    design_json : dict
        The V2 design dict (objects + assembly + job_name).
        NOT mutated; a deep copy is made first.
    edits : dict
        Mapping of dot-separated path -> new value.
        Example: {"objects.pyramid.dimensions.base_width_um": 12}
    strict : bool
        If True (default), reject type mismatches.
        If False, allow any value regardless of existing type.

    Returns
    -------
    modified : dict
        Deep copy of design_json with all edits applied.
    change_log : list[str]
        Human-readable list of changes made.

    Raises
    ------
    KeyError
        If a path does not exist in the design.
    TypeError
        If a new value's type is incompatible with the existing value
        and strict=True.
    ValueError
        If the edits dict is empty or the dot-path has fewer than 2 segments.
    """
    if not edits:
        raise ValueError("Edit dict is empty. Provide at least one key-value pair.")

    modified = copy.deepcopy(design_json)
    change_log: List[str] = []

    for path, new_value in edits.items():
        parts = path.split(".")
        if len(parts) < 2:
            raise ValueError(
                f"Dot-path '{path}' has only one segment. "
                f"Paths must be at least 'parent.key' (two levels deep)."
            )

        try:
            parent, key = _traverse(modified, parts)
        except KeyError as exc:
            raise KeyError(
                f"Invalid path '{path}': {exc}"
            ) from exc
        except TypeError as exc:
            raise TypeError(
                f"Cannot traverse path '{path}': {exc}"
            ) from exc

        existing_value = parent[key]

        if strict:
            _check_type_compatibility(existing_value, new_value, path)

        parent[key] = new_value

        change_log.append(
            f"EDIT | {path}: {existing_value!r} -> {new_value!r}"
        )
        print(f"[DotPathEditor] {change_log[-1]}")

    print(f"[DotPathEditor] Applied {len(change_log)} edits successfully.")
    return modified, change_log


def validate_edit_dict(edits: Dict[str, Any]) -> List[str]:
    """
    Lightweight pre-flight check on an edit dict.

    Returns a list of validation errors (empty = valid).
    Does NOT modify anything.
    """
    errors: List[str] = []
    for path, value in edits.items():
        parts = path.split(".")
        if len(parts) < 2:
            errors.append(f"Path '{path}' must contain at least one dot.")
        if not isinstance(path, str) or not path.strip():
            errors.append(f"Path must be a non-empty string, got: {path!r}")
        if value is None:
            errors.append(
                f"Value for '{path}' is None. "
                f"Use explicit 0 / '' / False instead of None."
            )
    return errors
