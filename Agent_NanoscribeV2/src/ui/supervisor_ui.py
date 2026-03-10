"""
supervisor_ui.py - Nanoscribe Design Agent UI

Streamlit interface for the full fabrication workflow:

  LIBRARY        - browse / load saved designs; start fresh or open a STEP file
  DESIGN         - prompt-driven design generation
  REDESIGN       - natural-language geometry edits + slice view + GWL export
  CHARACTERIZE   - post-fabrication SEM analysis → deviation report

Run:
    streamlit run Agent_NanoscribeV2/src/ui/supervisor_ui.py

On first launch a design library JSON is created at Outputs/_library.json
cataloguing every completed iteration folder found on disk.
"""

from __future__ import annotations

import base64
import importlib.util
import json
import shutil
import sys
import tempfile
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import streamlit as st

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------
_UI_DIR        = Path(__file__).resolve().parent
_SRC_DIR       = _UI_DIR.parent
_PROJECT_ROOT  = _SRC_DIR.parent
_CLASSIFICATION = _PROJECT_ROOT.parent.parent / "Agent-Nanoscribe_Classification"

for _p in [str(_SRC_DIR), str(_SRC_DIR / "supervisor"),
           str(_CLASSIFICATION / "src")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _load_mod(name: str, filepath: Path):
    spec = importlib.util.spec_from_file_location(name, str(filepath))
    mod  = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_fab_mod = _load_mod("fabrication_job", _SRC_DIR / "supervisor" / "fabrication_job.py")
_sup_mod = _load_mod("supervisor_module", _SRC_DIR / "supervisor" / "supervisor.py")

FabricationJob = _fab_mod.FabricationJob
Supervisor     = _sup_mod.Supervisor

import gwl_serializer          # noqa: E402  (already on sys.path)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PHASES = ["LIBRARY", "DESIGN", "REDESIGN", "CHARACTERIZE"]

PHASE_LABELS = {
    "LIBRARY":     "Library",
    "DESIGN":      "Design",
    "REDESIGN":    "Fabricate",
    "CHARACTERIZE": "Characterize",
}

OUTPUTS_ROOT  = _PROJECT_ROOT / "Outputs"
LIBRARY_FILE  = OUTPUTS_ROOT / "_library.json"

# ---------------------------------------------------------------------------
# Page config  (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Nanoscribe Design Agent",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Design Library helpers
# ---------------------------------------------------------------------------

def _scan_library() -> List[Dict]:
    """
    Walk Outputs/ and return one entry per completed iteration folder.
    Reads design.json for job name + prompt; counts iterations.
    """
    entries: List[Dict] = []
    if not OUTPUTS_ROOT.exists():
        return entries

    for job_dir in sorted(OUTPUTS_ROOT.iterdir()):
        if not job_dir.is_dir() or job_dir.name.startswith("_"):
            continue
        iter_dirs = sorted(job_dir.glob("iter_*"))
        if not iter_dirs:
            continue

        # Read design.json from the latest iter that has one
        design_meta: Dict = {}
        source = "prompt"
        for idir in reversed(iter_dirs):
            dp = idir / "design.json"
            if dp.exists():
                try:
                    with dp.open() as fh:
                        d = json.load(fh)
                    design_meta = d.get("metadata", {})
                    # Detect STEP-sourced designs
                    if d.get("metadata", {}).get("source") == "step_pipeline":
                        source = "step"
                    break
                except Exception:
                    pass

        entries.append({
            "job_id":     job_dir.name,
            "job_dir":    str(job_dir),
            "iterations": len(iter_dirs),
            "source":     source,
            "prompt":     design_meta.get("prompt", ""),
            "timestamp":  design_meta.get("timestamp", ""),
        })

    return entries


def _init_library():
    """Create _library.json on first startup; refresh it."""
    OUTPUTS_ROOT.mkdir(parents=True, exist_ok=True)
    entries = _scan_library()
    with LIBRARY_FILE.open("w") as fh:
        json.dump({"updated": datetime.now().isoformat(), "designs": entries}, fh, indent=2)
    return entries


def _load_library() -> List[Dict]:
    if not LIBRARY_FILE.exists():
        return _init_library()
    try:
        with LIBRARY_FILE.open() as fh:
            data = json.load(fh)
        return data.get("designs", [])
    except Exception:
        return _init_library()


def _save_to_library(entry: Dict):
    """Upsert one design entry into the library file."""
    entries = _load_library()
    existing = {e["job_id"] for e in entries}
    if entry["job_id"] not in existing:
        entries.insert(0, entry)
    else:
        entries = [entry if e["job_id"] == entry["job_id"] else e for e in entries]
    with LIBRARY_FILE.open("w") as fh:
        json.dump({"updated": datetime.now().isoformat(), "designs": entries}, fh, indent=2)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

def _init_session():
    defaults = {
        "phase":          "LIBRARY",
        "job":            None,
        "supervisor":     None,
        "error":          None,
        "status_log":     [],
        "prompt_history": [],
        "origin_prompt":  None,
        "clear_prompt":   False,
        "prompt_counter": 0,
        "library":        None,   # cached list of designs
        "step_output_json": None, # output.json from step_pipeline (for STEP path)
        "step_design_json": None, # synthetic design.json from step_pipeline
        "step_job_name":    None,
        # Characterization
        "char_expected":   None,  # expected_features.json content
        "char_observed":   None,  # validated observed_features dict
        "char_report":     None,  # deviation_report dict
        "char_sem_top":    None,  # bytes of sem_top.png
        "char_sem_oblique": None, # bytes of sem_oblique.png
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    if st.session_state.supervisor is None:
        try:
            st.session_state.supervisor = Supervisor(outputs_root=OUTPUTS_ROOT)
        except Exception as exc:
            st.session_state.error = f"Supervisor init failed: {exc}"

    if st.session_state.library is None:
        st.session_state.library = _load_library()


_init_session()


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def _log(msg: str):
    st.session_state.status_log.append(msg)


def _outputs_root() -> Path:
    return OUTPUTS_ROOT


def _full_reset(go_to_library: bool = True):
    job = st.session_state.job
    if job is not None:
        jdir = _outputs_root() / job.job_id
        if jdir.exists():
            shutil.rmtree(jdir, ignore_errors=True)
    keys_to_clear = [
        "job", "error", "status_log", "prompt_history",
        "origin_prompt", "step_output_json", "step_design_json", "step_job_name",
        "char_expected", "char_observed", "char_report",
        "char_sem_top", "char_sem_oblique",
    ]
    for k in keys_to_clear:
        st.session_state[k] = None if k != "status_log" and k != "prompt_history" else []
    st.session_state.clear_prompt = True
    st.session_state.phase = "LIBRARY" if go_to_library else "DESIGN"
    st.session_state.library = _load_library()


def _delete_iter(job: FabricationJob, iteration: int):
    idir = _outputs_root() / job.job_id / f"iter_{iteration:03d}"
    if idir.exists():
        shutil.rmtree(idir, ignore_errors=True)
        _log(f"Deleted iter_{iteration:03d}")


def _step_back():
    phase = st.session_state.phase
    job: FabricationJob = st.session_state.job

    if phase == "DESIGN":
        st.session_state.phase = "LIBRARY"
    elif phase == "REDESIGN":
        if job is None or job.iteration <= 1:
            _full_reset()
        else:
            last = job.iteration - 1
            _delete_iter(job, last)
            job.iteration = last
            prev = last - 1
            prev_dir = _outputs_root() / job.job_id / f"iter_{prev:03d}"
            if prev_dir.exists():
                for fname, attr in [
                    ("design.json",          "design_json"),
                    ("reduced.json",         "reduced_json"),
                    ("enriched_reduced.json","enriched_json"),
                    ("output.json",          "output_json"),
                ]:
                    fp = prev_dir / fname
                    if fp.exists():
                        with fp.open() as fh:
                            setattr(job, attr, json.load(fh))
                gwl_d = prev_dir / "gwl"
                job.gwl_path = gwl_d if gwl_d.exists() else None
            if job.metrics_history:
                job.metrics_history.pop()
            if job.iteration_metadata:
                job.iteration_metadata.pop()
            if st.session_state.prompt_history:
                st.session_state.prompt_history.pop()
            _log(f"Reverted to iter_{prev:03d}")
            st.session_state.phase = "REDESIGN"
    elif phase == "CHARACTERIZE":
        st.session_state.phase = "REDESIGN"


# ---------------------------------------------------------------------------
# Visualization helpers  (same as before)
# ---------------------------------------------------------------------------

def _render_primitives_3d(primitives: List[Dict], title: str = "Geometry Preview"):
    import matplotlib.pyplot as plt
    import numpy as np
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#fafafa")
    fig.patch.set_facecolor("#fafafa")

    for prim in primitives:
        ptype  = prim.get("type", "box")
        center = prim.get("center", [0, 0, 0])
        dims   = prim.get("dimensions", {})
        cx, cy, cz = center

        if ptype == "box":
            hx = dims.get("x_um", dims.get("length_um", dims.get("width_um", 1))) / 2
            hy = dims.get("y_um", dims.get("depth_um", 1)) / 2
            hz = dims.get("z_um", dims.get("height_um", 1)) / 2
            corners = np.array([
                [cx-hx,cy-hy,cz-hz],[cx+hx,cy-hy,cz-hz],
                [cx+hx,cy+hy,cz-hz],[cx-hx,cy+hy,cz-hz],
                [cx-hx,cy-hy,cz+hz],[cx+hx,cy-hy,cz+hz],
                [cx+hx,cy+hy,cz+hz],[cx-hx,cy+hy,cz+hz],
            ])
            rot = prim.get("rotation_z_deg", 0)
            if rot:
                rad = np.radians(rot)
                cos_a, sin_a = np.cos(rad), np.sin(rad)
                for i in range(8):
                    dx, dy = corners[i,0]-cx, corners[i,1]-cy
                    corners[i,0] = cx + dx*cos_a - dy*sin_a
                    corners[i,1] = cy + dx*sin_a + dy*cos_a
            faces = [
                [corners[0],corners[1],corners[2],corners[3]],
                [corners[4],corners[5],corners[6],corners[7]],
                [corners[0],corners[1],corners[5],corners[4]],
                [corners[2],corners[3],corners[7],corners[6]],
                [corners[0],corners[3],corners[7],corners[4]],
                [corners[1],corners[2],corners[6],corners[5]],
            ]
            ax.add_collection3d(Poly3DCollection(
                faces, alpha=0.22, facecolor="#4a90d9",
                edgecolor="#1a3a5c", linewidth=0.5))

        elif ptype == "cylinder":
            r = dims.get("diameter_um", 1) / 2
            h = dims.get("height_um", dims.get("z_um", 1))
            z_b, z_t = cz - h/2, cz + h/2
            theta = np.linspace(0, 2*np.pi, 24)
            xr = cx + r*np.cos(theta)
            yr = cy + r*np.sin(theta)
            for i in range(len(theta)-1):
                verts = [[xr[i],yr[i],z_b],[xr[i+1],yr[i+1],z_b],
                         [xr[i+1],yr[i+1],z_t],[xr[i],yr[i],z_t]]
                ax.add_collection3d(Poly3DCollection(
                    [verts], alpha=0.22, facecolor="#d97b4a",
                    edgecolor="#5c2a1a", linewidth=0.3))
            for z_cap in (z_b, z_t):
                cap = [[xr[i],yr[i],z_cap] for i in range(len(theta))]
                ax.add_collection3d(Poly3DCollection(
                    [cap], alpha=0.25, facecolor="#d97b4a",
                    edgecolor="#5c2a1a", linewidth=0.3))

    if primitives:
        all_c = [p.get("center",[0,0,0]) for p in primitives]
        xs, ys, zs = [c[0] for c in all_c],[c[1] for c in all_c],[c[2] for c in all_c]
        max_dim = max((abs(v) for p in primitives
                       for v in p.get("dimensions",{}).values()
                       if isinstance(v,(int,float))), default=1.0)
        m = max_dim * 0.6
        ax.set_xlim(min(xs)-m, max(xs)+m)
        ax.set_ylim(min(ys)-m, max(ys)+m)
        ax.set_zlim(min(zs)-m, max(zs)+m)

    ax.set_xlabel("X (µm)", fontsize=9, labelpad=8)
    ax.set_ylabel("Y (µm)", fontsize=9, labelpad=8)
    ax.set_zlabel("Z (µm)", fontsize=9, labelpad=8)
    ax.tick_params(labelsize=8)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=12)
    return fig


def _render_slice_view(output_json: Dict, layer_idx: int):
    import matplotlib.pyplot as plt

    layers = output_json.get("layers", [])
    if not layers or layer_idx >= len(layers):
        return None

    layer    = layers[layer_idx]
    segments = layer.get("segments", [])
    z_val    = layer.get("z_um", 0)

    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor("#fafafa")
    ax.set_facecolor("#fafafa")

    for seg in segments:
        sx, sy = seg["start"]
        ex, ey = seg["end"]
        if seg.get("type") == "contour":
            ax.plot([sx,ex],[sy,ey], color="#1a3a5c", linewidth=1.0, alpha=0.85)
        else:
            ax.plot([sx,ex],[sy,ey], color="#2e7d32", linewidth=0.5, alpha=0.7)

    ax.set_aspect("equal")
    ax.set_xlabel("X (µm)", fontsize=9)
    ax.set_ylabel("Y (µm)", fontsize=9)
    ax.set_title(
        f"Layer {layer_idx}  |  z = {z_val:.3f} µm  |  {len(segments)} segments",
        fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.15, linewidth=0.5)
    ax.tick_params(labelsize=8)
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([0],[0], color="#1a3a5c", linewidth=1.0, label="Contour"),
        Line2D([0],[0], color="#2e7d32", linewidth=0.5, label="Hatch"),
    ], loc="upper right", fontsize=8, framealpha=0.9)
    return fig


def _show_render_images(iter_dir: Path):
    """Display render PNGs from Renders/ if they exist."""
    render_dir = iter_dir / "Renders"
    if not render_dir.exists():
        return
    pngs = sorted(render_dir.rglob("*.png"))
    if not pngs:
        return
    st.markdown("**Renders**")
    # Group by subfolder
    groups: Dict[str, List[Path]] = {}
    for p in pngs:
        grp = p.parent.name
        groups.setdefault(grp, []).append(p)
    for grp, imgs in groups.items():
        with st.expander(grp, expanded=(grp == "final_assembly")):
            cols = st.columns(min(len(imgs), 3))
            for col, img in zip(cols, imgs):
                with col:
                    st.image(str(img), caption=img.name, use_container_width=True)


# ---------------------------------------------------------------------------
# NL parsers  (unchanged from original)
# ---------------------------------------------------------------------------

def _parse_redesign_prompt(prompt: str, design_json: Dict) -> Optional[Dict]:
    import re

    edits: Dict = {}
    objects  = design_json.get("objects", {})
    assembly = design_json.get("assembly", {})
    prompt_l = prompt.lower()

    dim_kw = {
        "width": "base_width_um", "base_width": "base_width_um",
        "height": "height_um", "depth": "depth_um", "length": "length_um",
        "diameter": "diameter_um", "radius": "radius_um",
        "x_um": "x_um", "y_um": "y_um", "z_um": "z_um",
        "top_width": "top_width_um", "top_diameter": "top_diameter_um",
    }

    def _component_height(comp: Dict) -> Optional[float]:
        d = comp.get("dimensions", {})
        h = d.get("height_um", d.get("z_um"))
        if isinstance(h, (int, float)):
            return float(h)
        return None

    def _stack_touch_edits(obj_name: str, comps: List[Dict], target_total_height: Optional[float] = None):
        if len(comps) < 2:
            return

        stack = []
        for i, comp in enumerate(comps):
            center = comp.get("center", [])
            h = _component_height(comp)
            if len(center) < 3 or h is None:
                return
            stack.append((i, float(center[2]), h))

        stack.sort(key=lambda t: t[1])
        idxs = [t[0] for t in stack]
        heights = [t[2] for t in stack]

        min_z = min(cz - h / 2.0 for _, cz, h in stack)

        if target_total_height is not None:
            fixed_sum = sum(heights[:-1])
            heights[-1] = max(0.01, float(target_total_height) - fixed_sum)
            top_idx = idxs[-1]
            edits[f"objects.{obj_name}.components.{top_idx}.dimensions.height_um"] = round(heights[-1], 6)

        running = min_z
        for idx, h in zip(idxs, heights):
            new_cz = running + h / 2.0
            edits[f"objects.{obj_name}.components.{idx}.center.2"] = round(new_cz, 6)
            running += h

    set_pat = re.compile(
        r"(?:set|change|make|increase|decrease|reduce|adjust)\s+"
        r"(?:the\s+)?(?:(?:pyramid|cone|cylinder|box|object)\s+)?"
        r"([\w\s]+?)\s+to\s+([\d.]+)", re.IGNORECASE)
    for m in set_pat.finditer(prompt):
        kw = m.group(1).strip().lower()
        val = float(m.group(2))
        dk = next((v for k, v in dim_kw.items() if k in kw), None)
        if dk:
            for oname, odef in objects.items():
                if odef.get("type") != "geometry":
                    continue
                for i, comp in enumerate(odef.get("components", [])):
                    if dk in comp.get("dimensions", {}):
                        edits[f"objects.{oname}.components.{i}.dimensions.{dk}"] = val
                        break
                    if dk in comp.get("construction", {}):
                        edits[f"objects.{oname}.components.{i}.construction.{dk}"] = val
                        break

    arr_pat = re.compile(r"(\d+)\s*(?:x|by)\s*(\d+)\s*(?:array|grid)?", re.IGNORECASE)
    for m in arr_pat.finditer(prompt_l):
        nx, ny = int(m.group(1)), int(m.group(2))
        if "grid" in assembly:
            edits["assembly.grid.x"] = nx
            edits["assembly.grid.y"] = ny

    delta_pat = re.compile(
        r"(increase|decrease)\s+(?:the\s+)?(?:(?:pyramid|cone|cylinder|box)\s+)?"
        r"([\w\s]+?)\s+(?:by|to)\s+([\d.]+)", re.IGNORECASE)
    for m in delta_pat.finditer(prompt):
        action = m.group(1).lower()
        kw     = m.group(2).strip().lower()
        val    = float(m.group(3))
        dk = next((v for k, v in dim_kw.items() if k in kw), None)
        if dk:
            for oname, odef in objects.items():
                if odef.get("type") != "geometry":
                    continue
                for i, comp in enumerate(odef.get("components", [])):
                    cur = comp.get("dimensions", {}).get(dk)
                    if cur is not None:
                        nv = cur + val if action == "increase" else cur - val
                        edits[f"objects.{oname}.components.{i}.dimensions.{dk}"] = round(nv, 6)
                        break

    layer_pat = re.compile(r"(\d+)\s+layers", re.IGNORECASE)
    for m in layer_pat.finditer(prompt):
        nl = int(m.group(1))
        for oname, odef in objects.items():
            if odef.get("type") != "geometry":
                continue
            for i, comp in enumerate(odef.get("components", [])):
                if "construction" in comp and "layers" in comp["construction"]:
                    edits[f"objects.{oname}.components.{i}.construction.layers"] = nl

    total_h_pat = re.compile(
        r"(?:total|overall)\s+height(?:\s+of\s+(?:the\s+)?object)?\s*(?:to|=)?\s*([\d.]+)",
        re.IGNORECASE,
    )
    m_total = total_h_pat.search(prompt)
    if m_total:
        target_h = float(m_total.group(1))
        for oname, odef in objects.items():
            if odef.get("type") == "geometry":
                _stack_touch_edits(oname, odef.get("components", []), target_total_height=target_h)

    if any(w in prompt_l for w in ("touch", "touching", "no gap", "flush", "stacked")):
        for oname, odef in objects.items():
            if odef.get("type") == "geometry":
                _stack_touch_edits(oname, odef.get("components", []), target_total_height=None)

    return edits or None


def _parse_override_prompt(prompt: str) -> Optional[Dict]:
    import re
    overrides: Dict = {}
    param_kw = {
        "slice_spacing": "slice_spacing_um", "slice spacing": "slice_spacing_um",
        "hatch_spacing": "hatch_spacing_um", "hatch spacing": "hatch_spacing_um",
        "hatch": "hatch_spacing_um",
        "contour_count": "solid_contour_count", "contour count": "solid_contour_count",
        "contours": "solid_contour_count",
    }

    set_pat = re.compile(
        r"(?:set|change|make)\s+(?:the\s+)?([\w\s]+?)\s+to\s+([\d.]+)", re.IGNORECASE)
    for m in set_pat.finditer(prompt):
        kw  = m.group(1).strip().lower()
        val = float(m.group(2))
        pn  = next((v for k, v in param_kw.items() if k in kw), None)
        if pn:
            overrides[pn] = val

    delta_pat = re.compile(
        r"(increase|decrease|reduce)\s+(?:the\s+)?([\w\s]+?)\s+by\s+([\d.]+)", re.IGNORECASE)
    for m in delta_pat.finditer(prompt):
        action = m.group(1).lower()
        kw     = m.group(2).strip().lower()
        val    = float(m.group(3))
        pn     = next((v for k, v in param_kw.items() if k in kw), None)
        if pn:
            overrides[pn] = f"{'+'if action=='increase' else '-'}{val}"

    if "slightly" in prompt.lower() and not overrides:
        neg = any(w in prompt.lower() for w in ("reduce","decrease","smaller"))
        if "hatch"  in prompt.lower(): overrides["hatch_spacing_um"]  = f"{'-' if neg else '+'}0.03"
        if "slice"  in prompt.lower(): overrides["slice_spacing_um"]  = f"{'-' if neg else '+'}0.05"

    return overrides or None


# ---------------------------------------------------------------------------
# Phase handlers
# ---------------------------------------------------------------------------

def _handle_design(prompt: str):
    sup: Supervisor  = st.session_state.supervisor
    job_id = prompt.strip().replace(" ", "_")[:40]
    job    = FabricationJob(job_id=job_id, base_prompt=prompt)
    st.session_state.job           = job
    st.session_state.origin_prompt = prompt
    _log(f"Generating design: {prompt}")
    try:
        sup.run_prompt_job(job)
        st.session_state.phase = "REDESIGN"
        _log(f"Design complete. {job.reduced_json['metadata']['num_primitives']} primitives.")
        st.session_state.prompt_history.append(("DESIGN", prompt))
        # Update library
        _save_to_library({
            "job_id":     job.job_id,
            "job_dir":    str(_outputs_root() / job.job_id),
            "iterations": job.iteration,
            "source":     "prompt",
            "prompt":     prompt,
            "timestamp":  datetime.now().isoformat(),
        })
        st.session_state.library = _load_library()
    except Exception as exc:
        st.session_state.error = f"Design generation failed:\n{traceback.format_exc()}"
        _log(f"ERROR: {exc}")
    st.session_state.clear_prompt = True


def _handle_step_upload(uploaded_file, hatch: float, slice_sp: float, contours: int):
    """Run the STEP pipeline for an uploaded .step file."""
    import step_pipeline

    # Write uploaded bytes to a temp file
    suffix = Path(uploaded_file.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = Path(tmp.name)

    job_name = Path(uploaded_file.name).stem
    _log(f"Running STEP pipeline: {uploaded_file.name}")

    try:
        out_dir = step_pipeline.run_step_pipeline(
            step_path         = tmp_path,
            job_name          = job_name,
            hatch_override    = hatch,
            slice_override    = slice_sp,
            contours_override = contours,
        )
        # Load produced output.json and design.json
        out_json_path    = out_dir / "output.json"
        design_json_path = out_dir / "design.json"

        if out_json_path.exists():
            with out_json_path.open() as fh:
                st.session_state.step_output_json = json.load(fh)
        if design_json_path.exists():
            with design_json_path.open() as fh:
                st.session_state.step_design_json = json.load(fh)

        st.session_state.step_job_name = job_name
        _log(f"STEP pipeline complete → {out_dir}")

        _save_to_library({
            "job_id":     job_name,
            "job_dir":    str(out_dir.parent),
            "iterations": 1,
            "source":     "step",
            "prompt":     f"STEP: {uploaded_file.name}",
            "timestamp":  datetime.now().isoformat(),
        })
        st.session_state.library = _load_library()

        # Move to REDESIGN view to show slices & renders
        st.session_state.phase = "REDESIGN"

    except Exception as exc:
        st.session_state.error = f"STEP pipeline failed:\n{traceback.format_exc()}"
        _log(f"ERROR: {exc}")
    finally:
        tmp_path.unlink(missing_ok=True)

    st.session_state.clear_prompt = True


def _handle_load_design(entry: Dict):
    """Load a saved design from the library back into session."""
    job_dir = Path(entry["job_dir"])
    # Find the latest iter
    iter_dirs = sorted(job_dir.glob("iter_*"))
    if not iter_dirs:
        st.session_state.error = "No iteration folders found in saved job."
        return

    latest_iter = iter_dirs[-1]

    # Check if this is a STEP-sourced job
    if entry.get("source") == "step":
        out_p = latest_iter / "output.json"
        des_p = latest_iter / "design.json"
        if out_p.exists():
            with out_p.open() as fh:
                st.session_state.step_output_json = json.load(fh)
        if des_p.exists():
            with des_p.open() as fh:
                st.session_state.step_design_json = json.load(fh)
        st.session_state.step_job_name = entry["job_id"]
        st.session_state.phase = "REDESIGN"
        _log(f"Loaded STEP job: {entry['job_id']}")
        return

    # Prompt-sourced: reconstruct a FabricationJob from disk
    job = FabricationJob(
        job_id      = entry["job_id"],
        base_prompt = entry.get("prompt", ""),
        iteration   = len(iter_dirs),
    )
    for fname, attr in [
        ("design.json",           "design_json"),
        ("reduced.json",          "reduced_json"),
        ("enriched_reduced.json", "enriched_json"),
        ("output.json",           "output_json"),
    ]:
        fp = latest_iter / fname
        if fp.exists():
            with fp.open() as fh:
                setattr(job, attr, json.load(fh))
    gwl_d = latest_iter / "GWL"
    job.gwl_path = gwl_d if gwl_d.exists() else None

    st.session_state.job           = job
    st.session_state.origin_prompt = entry.get("prompt", "")
    st.session_state.phase         = "REDESIGN"
    _log(f"Loaded job: {entry['job_id']} (iter {job.iteration})")


def _handle_redesign(prompt: str):
    job: FabricationJob = st.session_state.job
    sup: Supervisor     = st.session_state.supervisor

    edits = _parse_redesign_prompt(prompt, job.design_json)
    if not edits:
        st.session_state.error = (
            "Could not parse redesign. Try:\n"
            "  'set height to 15'\n"
            "  'make the array 8 by 8'\n"
            "  'increase width by 2'")
        return
    _log(f"Edits: {json.dumps(edits, indent=2)}")
    try:
        sup.apply_redesign(job, edits)
        _log(f"Redesign applied. Iteration {job.iteration - 1}.")
        st.session_state.prompt_history.append(("REDESIGN", prompt))
    except Exception as exc:
        st.session_state.error = f"Redesign failed:\n{traceback.format_exc()}"
        _log(f"ERROR: {exc}")
    st.session_state.clear_prompt = True


def _handle_override(prompt: str):
    job: FabricationJob = st.session_state.job
    sup: Supervisor     = st.session_state.supervisor

    overrides = _parse_override_prompt(prompt)
    if not overrides:
        st.session_state.error = (
            "Could not parse override. Try:\n"
            "  'increase slice spacing by 0.1'\n"
            "  'set hatch spacing to 0.3'")
        return
    _log(f"Overrides: {json.dumps(overrides, indent=2)}")
    try:
        sup.apply_overrides(job, overrides)
        _log(f"Overrides applied. Iteration {job.iteration - 1}.")
        st.session_state.prompt_history.append(("OVERRIDE", prompt))
    except Exception as exc:
        st.session_state.error = f"Override failed:\n{traceback.format_exc()}"
        _log(f"ERROR: {exc}")
    st.session_state.clear_prompt = True


# ---------------------------------------------------------------------------
# Characterization helpers
# ---------------------------------------------------------------------------

def _handle_run_deviation(iter_dir: Path):
    """Run the deviation agent using files in iter_dir."""
    from inspection.run_deviation_agent import run_deviation_agent
    from inspection.extract_expected_features import extract_expected_features
    from inspection.load_observed_features import validate_observed_features

    # Expected features
    enriched_path = iter_dir / "enriched_reduced.json"
    if not enriched_path.exists():
        st.session_state.error = "enriched_reduced.json not found in this iteration."
        return
    with enriched_path.open() as fh:
        enriched = json.load(fh)
    expected = extract_expected_features(enriched)
    st.session_state.char_expected = expected

    # Observed features
    observed = st.session_state.char_observed
    if observed is None:
        st.session_state.error = "Upload observed_features.json or fill it in first."
        return

    # SEM images
    sem_top_bytes     = st.session_state.char_sem_top
    sem_oblique_bytes = st.session_state.char_sem_oblique
    if not sem_top_bytes or not sem_oblique_bytes:
        st.session_state.error = "Upload both sem_top.png and sem_oblique.png."
        return

    # Write images to temp files for the agent
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as t1:
        t1.write(sem_top_bytes)
        top_path = Path(t1.name)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as t2:
        t2.write(sem_oblique_bytes)
        oblique_path = Path(t2.name)

    try:
        _log("Running deviation agent...")
        report = run_deviation_agent(
            expected_features = expected,
            observed_features = observed,
            sem_top_path      = top_path,
            sem_oblique_path  = oblique_path,
        )
        st.session_state.char_report = report

        # Also write report into the iteration folder's analysis/ dir
        analysis_dir = iter_dir / "analysis"
        analysis_dir.mkdir(exist_ok=True)
        with (analysis_dir / "expected_features.json").open("w") as fh:
            json.dump(expected, fh, indent=2)
        with (analysis_dir / "deviation_report.json").open("w") as fh:
            json.dump(report, fh, indent=2)
        _log("Deviation report written to analysis/")

    except Exception as exc:
        st.session_state.error = f"Deviation agent failed:\n{traceback.format_exc()}"
        _log(f"ERROR: {exc}")
    finally:
        top_path.unlink(missing_ok=True)
        oblique_path.unlink(missing_ok=True)


def _show_deviation_report(report: Dict):
    """Render a deviation report in structured Streamlit components."""
    sc = report.get("structure_check", {})
    match = sc.get("structure_match", False)

    if match:
        st.success("Structure match: PASS")
    else:
        st.error("Structure match: FAIL")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Expected**")
        st.info(sc.get("expected_structure", "—"))
    with col2:
        st.markdown("**Observed**")
        st.warning(sc.get("observed_structure", "—"))

    st.divider()
    feature_checks = report.get("feature_checks", [])
    if feature_checks:
        st.markdown(f"**Feature checks ({len(feature_checks)})**")
        for fc in feature_checks:
            dev = fc.get("deviation_type", "none")
            conf = fc.get("confidence", "")
            icon = "✅" if dev == "none" else "⚠️"
            with st.expander(f"{icon} {fc.get('feature_id','?')} — {dev} [{conf}]"):
                st.text(f"Expected type : {fc.get('expected_type','')}")
                st.text(f"Observed state: {fc.get('observed_state','')}")

    failures = report.get("failures", [])
    if failures:
        st.divider()
        st.markdown(f"**Failures ({len(failures)})**")
        for f in failures:
            st.error(
                f"[{f.get('failure_type','?')}] "
                f"feature={f.get('feature_id','?')} — {f.get('description','')}"
            )
    else:
        st.success("No failures detected.")


# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------

def main():
    import matplotlib.pyplot as plt

    phase = st.session_state.phase
    job: Optional[FabricationJob] = st.session_state.job

    # ── Phase nav bar ────────────────────────────────────────────────────────
    nav_cols = st.columns(len(PHASES))
    for i, p in enumerate(PHASES):
        with nav_cols[i]:
            label = PHASE_LABELS[p]
            if p == phase:
                st.markdown(f"**› {label.upper()}**")
            else:
                if st.button(label, key=f"nav_{p}", use_container_width=True):
                    # Allow forward-only navigation where data exists
                    if p == "LIBRARY":
                        st.session_state.phase = "LIBRARY"
                        st.rerun()
                    elif p == "CHARACTERIZE" and (
                        job is not None or st.session_state.step_output_json is not None):
                        st.session_state.phase = "CHARACTERIZE"
                        st.rerun()

    st.divider()

    # ── Error banner ─────────────────────────────────────────────────────────
    if st.session_state.error:
        st.error(st.session_state.error)
        st.session_state.error = None

    # ======================================================================
    # LIBRARY phase
    # ======================================================================
    if phase == "LIBRARY":
        st.markdown("## Design Library")

        lib = st.session_state.library or []

        refresh_col, new_col, step_col = st.columns([1, 1, 1])
        with refresh_col:
            if st.button("↻ Refresh", use_container_width=True):
                st.session_state.library = _load_library()
                st.rerun()
        with new_col:
            if st.button("＋ New Prompt Design", type="primary", use_container_width=True):
                st.session_state.phase = "DESIGN"
                st.rerun()
        with step_col:
            if st.button("＋ Import STEP File", use_container_width=True):
                st.session_state.phase = "DESIGN"
                st.session_state["_step_mode"] = True
                st.rerun()

        if not lib:
            st.info("No saved designs yet. Create one above.")
        else:
            st.markdown(f"**{len(lib)} saved design(s)**")
            for entry in lib:
                source_badge = "📄 STEP" if entry.get("source") == "step" else "💬 Prompt"
                with st.expander(
                    f"{source_badge}  {entry['job_id']}  "
                    f"({entry.get('iterations', 1)} iter)  "
                    f"— {entry.get('timestamp','')[:16]}"
                ):
                    if entry.get("prompt"):
                        st.caption(entry["prompt"][:120])

                    col_load, col_char, col_del = st.columns(3)
                    with col_load:
                        if st.button("Load", key=f"load_{entry['job_id']}",
                                     use_container_width=True):
                            _handle_load_design(entry)
                            st.rerun()
                    with col_char:
                        if st.button("Characterize", key=f"char_{entry['job_id']}",
                                     use_container_width=True):
                            _handle_load_design(entry)
                            st.session_state.phase = "CHARACTERIZE"
                            st.rerun()
                    with col_del:
                        if st.button("Delete", key=f"del_{entry['job_id']}",
                                     use_container_width=True):
                            jdir = Path(entry["job_dir"])
                            if jdir.exists():
                                shutil.rmtree(jdir, ignore_errors=True)
                            st.session_state.library = _load_library()
                            _log(f"Deleted {entry['job_id']}")
                            st.rerun()

        return  # LIBRARY phase done

    # ======================================================================
    # DESIGN phase
    # ======================================================================
    if phase == "DESIGN":
        step_mode = st.session_state.get("_step_mode", False)

        left, right = st.columns([1, 2])
        with left:
            if step_mode:
                st.markdown("### Import STEP File")
                uploaded = st.file_uploader(
                    "Upload .step / .stp file",
                    type=["step", "stp"],
                    key="step_uploader",
                )
                st.markdown("**Slicing parameters** *(override PrintParameters.txt)*")
                hatch    = st.number_input("Hatch spacing (µm)", value=0.35, step=0.05, format="%.3f")
                slice_sp = st.number_input("Slice spacing (µm)", value=0.60, step=0.05, format="%.3f")
                contours = st.number_input("Contour passes",     value=2,    step=1,    format="%d")

                c1, c2 = st.columns(2)
                with c1:
                    if st.button("Run STEP Pipeline", type="primary", use_container_width=True):
                        if uploaded:
                            with st.spinner("Slicing STEP geometry…"):
                                _handle_step_upload(uploaded, hatch, slice_sp, int(contours))
                            st.rerun()
                        else:
                            st.session_state.error = "Upload a STEP file first."
                with c2:
                    if st.button("← Back to Library", use_container_width=True):
                        st.session_state["_step_mode"] = False
                        st.session_state.phase = "LIBRARY"
                        st.rerun()
            else:
                st.markdown("### New Design")
                if st.session_state.clear_prompt:
                    st.session_state.prompt_counter += 1
                    st.session_state.clear_prompt = False

                prompt = st.text_area(
                    "Describe the structure to fabricate",
                    placeholder="e.g. 5×5 array of cylinders, 2 µm diameter, 5 µm tall",
                    height=120,
                    key=f"prompt_design_{st.session_state.prompt_counter}",
                )

                c1, c2 = st.columns(2)
                with c1:
                    if st.button("Generate", type="primary", use_container_width=True):
                        if prompt.strip():
                            with st.spinner("Generating design…"):
                                _handle_design(prompt.strip())
                            st.rerun()
                        else:
                            st.session_state.error = "Enter a design prompt."
                with c2:
                    if st.button("← Library", use_container_width=True):
                        st.session_state.phase = "LIBRARY"
                        st.rerun()

        with right:
            st.caption("Enter a prompt or upload a STEP file to begin.")

        return  # DESIGN phase done

    # ======================================================================
    # REDESIGN / Fabricate phase
    # ======================================================================
    if phase == "REDESIGN":
        # Determine active output_json (STEP path or prompt path)
        is_step = (st.session_state.step_output_json is not None
                   and st.session_state.job is None)
        active_output_json = (st.session_state.step_output_json if is_step
                              else (job.output_json if job else None))
        active_design_json = (st.session_state.step_design_json if is_step
                              else (job.design_json if job else None))
        active_reduced     = None if is_step else (job.reduced_json if job else None)

        # Determine current iter_dir for render lookup
        iter_dir: Optional[Path] = None
        if is_step:
            jname = st.session_state.step_job_name
            if jname:
                candidate = _outputs_root() / jname / "iter_000"
                if candidate.exists():
                    iter_dir = candidate
        elif job:
            prev = max(0, job.iteration - 1)
            candidate = _outputs_root() / job.job_id / f"iter_{prev:03d}"
            if candidate.exists():
                iter_dir = candidate

        left, right = st.columns([1, 2])

        with left:
            if st.session_state.origin_prompt:
                st.info(st.session_state.origin_prompt)

            if is_step:
                st.success(f"STEP job loaded: **{st.session_state.step_job_name}**")
            else:
                # Prompt-path: show redesign + override inputs
                if st.session_state.clear_prompt:
                    st.session_state.prompt_counter += 1
                    st.session_state.clear_prompt = False

                prompt = st.text_area(
                    "Redesign / Parameter override",
                    placeholder=(
                        "Redesign: 'increase height to 20'\n"
                        "Override: 'reduce hatch spacing by 0.05'"
                    ),
                    height=100,
                    key=f"prompt_redesign_{st.session_state.prompt_counter}",
                    label_visibility="collapsed",
                )
                r1, r2 = st.columns(2)
                with r1:
                    if st.button("Apply Redesign", type="primary", use_container_width=True):
                        if prompt.strip():
                            with st.spinner("Applying redesign…"):
                                _handle_redesign(prompt.strip())
                            st.rerun()
                        else:
                            st.session_state.error = "Enter a redesign instruction."
                with r2:
                    if st.button("Apply Override", use_container_width=True):
                        if prompt.strip():
                            with st.spinner("Applying override…"):
                                _handle_override(prompt.strip())
                            st.rerun()
                        else:
                            st.session_state.error = "Enter a parameter override."

            st.divider()

            # Save button
            if st.button("💾 Save to Library", use_container_width=True):
                if is_step:
                    _save_to_library({
                        "job_id":     st.session_state.step_job_name,
                        "job_dir":    str(_outputs_root() / st.session_state.step_job_name),
                        "iterations": 1,
                        "source":     "step",
                        "prompt":     f"STEP: {st.session_state.step_job_name}",
                        "timestamp":  datetime.now().isoformat(),
                    })
                elif job:
                    _save_to_library({
                        "job_id":     job.job_id,
                        "job_dir":    str(_outputs_root() / job.job_id),
                        "iterations": job.iteration,
                        "source":     "prompt",
                        "prompt":     st.session_state.origin_prompt or "",
                        "timestamp":  datetime.now().isoformat(),
                    })
                st.session_state.library = _load_library()
                st.success("Saved to library.")

            # Characterize button
            if st.button("🔬 Enter Characterization", use_container_width=True):
                st.session_state.phase = "CHARACTERIZE"
                st.rerun()

            st.divider()
            b1, b2 = st.columns(2)
            with b1:
                if st.button("← Back", use_container_width=True):
                    _step_back()
                    st.rerun()
            with b2:
                if st.button("Reset", use_container_width=True):
                    _full_reset()
                    st.rerun()

            # Job info
            if job and not is_step:
                st.divider()
                st.markdown(f"**{job.job_id}** — iter {job.iteration}")
                if job.metrics_history:
                    for k, v in job.metrics_history[-1].items():
                        if k == "delta":
                            continue
                        val_text = f"{v:.4f}" if isinstance(v, float) else str(v)
                        st.text(f"  {k.replace('_',' ')}: "
                                f"{val_text}")
            if st.session_state.prompt_history:
                with st.expander("History"):
                    for tag, txt in reversed(st.session_state.prompt_history):
                        st.text(f"[{tag}] {txt}")
            if st.session_state.status_log:
                with st.expander("Log"):
                    for entry in reversed(st.session_state.status_log[-20:]):
                        st.text(entry)

        with right:
            # ── Tab: 3D / Slices / Renders ────────────────────────────────
            tab_labels = ["Slice View", "3D Preview", "Renders"]
            tabs = st.tabs(tab_labels)

            with tabs[0]:
                # Slice viewer
                if active_output_json and active_output_json.get("layers"):
                    layers    = active_output_json["layers"]
                    num_layers = len(layers)
                    st.caption(f"{num_layers} layers")
                    layer_idx = st.slider("Layer", 0, max(0, num_layers-1), 0, key="layer_slider")
                    try:
                        fig = _render_slice_view(active_output_json, layer_idx)
                        if fig:
                            st.pyplot(fig)
                            plt.close(fig)
                    except Exception as exc:
                        st.warning(f"Slice render error: {exc}")
                    with st.expander("Layer table"):
                        st.dataframe([
                            {"layer": i, "z_µm": round(l["z_um"], 4),
                             "segments": len(l.get("segments", []))}
                            for i, l in enumerate(layers)
                        ], use_container_width=True)
                else:
                    st.info("No slice data available.")

            with tabs[1]:
                # 3D preview (prompt path only)
                if active_reduced and active_reduced.get("primitives"):
                    prims = active_reduced["primitives"]
                    st.caption(f"{len(prims)} primitives")
                    try:
                        fig = _render_primitives_3d(prims)
                        st.pyplot(fig)
                        plt.close(fig)
                    except Exception as exc:
                        st.warning(f"3D render error: {exc}")
                elif is_step:
                    st.info("3D preview not available for STEP imports (no primitive model).")
                else:
                    st.info("No geometry loaded.")

            with tabs[2]:
                # Render images
                if iter_dir:
                    _show_render_images(iter_dir)
                else:
                    st.info("No renders found.")

        return  # REDESIGN phase done

    # ======================================================================
    # CHARACTERIZE phase
    # ======================================================================
    if phase == "CHARACTERIZE":
        st.markdown("## Characterization — SEM Deviation Analysis")

        # Determine which iter_dir to use for reading enriched_reduced.json
        is_step = (st.session_state.step_output_json is not None
                   and st.session_state.job is None)
        iter_dir: Optional[Path] = None
        if is_step:
            jname = st.session_state.step_job_name
            if jname:
                candidate = _outputs_root() / jname / "iter_000"
                if candidate.exists():
                    iter_dir = candidate
        elif job:
            prev = max(0, job.iteration - 1)
            candidate = _outputs_root() / job.job_id / f"iter_{prev:03d}"
            if candidate.exists():
                iter_dir = candidate

        left, right = st.columns([1, 2])

        with left:
            st.markdown("### Inputs")

            # ── SEM images ───────────────────────────────────────────────
            st.markdown("**SEM Images**")
            top_file = st.file_uploader(
                "sem_top.png (top-down view)", type=["png","tiff","tif","jpg","jpeg"],
                key="sem_top_upload")
            if top_file:
                st.session_state.char_sem_top = top_file.read()
                st.image(st.session_state.char_sem_top, caption="Top-down SEM",
                         use_container_width=True)

            oblique_file = st.file_uploader(
                "sem_oblique.png (oblique view)", type=["png","tiff","tif","jpg","jpeg"],
                key="sem_oblique_upload")
            if oblique_file:
                st.session_state.char_sem_oblique = oblique_file.read()
                st.image(st.session_state.char_sem_oblique, caption="Oblique SEM",
                         use_container_width=True)

            st.divider()

            # ── Observed features ─────────────────────────────────────────
            st.markdown("**Observed Features** *(from segmentation pipeline)*")
            obs_tab1, obs_tab2 = st.tabs(["Upload JSON", "Enter manually"])

            with obs_tab1:
                obs_file = st.file_uploader(
                    "observed_features.json",
                    type=["json"],
                    key="obs_upload",
                )
                if obs_file:
                    try:
                        from inspection.load_observed_features import validate_observed_features
                        obs_data = json.load(obs_file)
                        validate_observed_features(obs_data)
                        st.session_state.char_observed = obs_data
                        st.success("Observed features validated.")
                    except Exception as exc:
                        st.error(f"Validation error: {exc}")

            with obs_tab2:
                n_obj = st.number_input("objects_detected", value=1, min_value=0, step=1)
                n_hol = st.number_input("holes_detected",   value=0, min_value=0, step=1)
                area  = st.number_input("area (px²)",       value=4000.0, min_value=0.0)
                conv  = st.slider("convexity",  0.0, 1.0, 0.8)
                circ  = st.slider("circularity",0.0, 1.0, 0.7)
                if st.button("Use these values", use_container_width=True):
                    st.session_state.char_observed = {
                        "objects_detected": int(n_obj),
                        "holes_detected":   int(n_hol),
                        "shape_descriptors": {
                            "area": float(area),
                            "convexity":   float(conv),
                            "circularity": float(circ),
                        }
                    }
                    st.success("Observed features set.")

            st.divider()

            # ── Run button ────────────────────────────────────────────────
            ready = (st.session_state.char_sem_top is not None
                     and st.session_state.char_sem_oblique is not None
                     and st.session_state.char_observed is not None
                     and iter_dir is not None)

            if not ready:
                missing = []
                if not st.session_state.char_sem_top:     missing.append("sem_top")
                if not st.session_state.char_sem_oblique: missing.append("sem_oblique")
                if not st.session_state.char_observed:    missing.append("observed_features")
                if iter_dir is None:                       missing.append("fabrication job")
                st.warning(f"Still needed: {', '.join(missing)}")

            if st.button("▶ Run Deviation Analysis", type="primary",
                         use_container_width=True, disabled=not ready):
                with st.spinner("Calling deviation agent…"):
                    _handle_run_deviation(iter_dir)
                st.rerun()

            st.divider()
            if st.button("← Back to Fabricate", use_container_width=True):
                st.session_state.phase = "REDESIGN"
                st.rerun()

        with right:
            st.markdown("### Results")

            report = st.session_state.char_report

            if report is None:
                if iter_dir:
                    # Check if a previous report exists on disk
                    saved_report = iter_dir / "analysis" / "deviation_report.json"
                    if saved_report.exists():
                        with saved_report.open() as fh:
                            report = json.load(fh)
                        st.session_state.char_report = report
                        st.caption("Loaded previous deviation report from disk.")

            if report:
                _show_deviation_report(report)

                st.divider()
                # Show expected features
                if st.session_state.char_expected:
                    with st.expander("Expected Features"):
                        ef = st.session_state.char_expected
                        st.caption(f"{len(ef.get('objects',[]))} objects, "
                                   f"{len(ef.get('holes',[]))} holes")
                        st.json(ef)

                # Download report
                report_bytes = json.dumps(report, indent=2).encode()
                st.download_button(
                    "⬇ Download deviation_report.json",
                    data=report_bytes,
                    file_name="deviation_report.json",
                    mime="application/json",
                    use_container_width=True,
                )
            else:
                st.info("Run the deviation analysis to see results here.")

                # Show expected features preview if we can compute them
                if iter_dir and (iter_dir / "enriched_reduced.json").exists():
                    from inspection.extract_expected_features import extract_expected_features
                    with (iter_dir / "enriched_reduced.json").open() as fh:
                        enriched = json.load(fh)
                    ef = extract_expected_features(enriched)
                    st.markdown("**Expected geometry (preview)**")
                    st.caption(f"{len(ef['objects'])} objects, {len(ef['holes'])} holes")
                    with st.expander("View expected features"):
                        st.json(ef)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
else:
    main()
