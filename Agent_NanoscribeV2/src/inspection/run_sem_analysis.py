"""
run_sem_analysis.py - SEM Analysis Pipeline Runner

Top-level entry point for the SEM deviation analysis system.

This script runs the full SEM pipeline for a given iteration folder
without modifying any existing pipeline outputs.

READS (existing pipeline outputs — never modified):
    <iter_folder>/enriched_reduced.json

READS (provided by external segmentation system):
    <iter_folder>/inspection/observed_features.json
    <iter_folder>/inspection/sem_top.png
    <iter_folder>/inspection/sem_oblique.png

WRITES (new files only, inside iteration folder):
    <iter_folder>/analysis/expected_features.json
    <iter_folder>/analysis/deviation_report.json

Pipeline steps:
    1. Load enriched_reduced.json
    2. Generate expected_features.json   → [SEM] extracting expected features
    3. Load observed_features.json       → [SEM] loading observed features
    4. Load SEM images (validation)
    5. Run deviation agent               → [SEM] running deviation agent
    6. Write deviation_report.json       → [SEM] writing deviation report

Usage:
    python run_sem_analysis.py Outputs/JobName/iter_000/

    # Or with an absolute path:
    python run_sem_analysis.py C:/path/to/Outputs/JobName/iter_000/
"""

import json
import sys
from pathlib import Path


# Allow running as a standalone script (no package install required).
# Insert the src/ directory into sys.path so relative imports resolve.
_SRC_DIR = Path(__file__).parent.parent.resolve()
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from inspection.extract_expected_features import run as extract_expected
from inspection.load_observed_features    import run as load_observed
from inspection.run_deviation_agent       import run as run_agent


# ======================================================
# PREFLIGHT CHECKS
# ======================================================

def _check_pipeline_outputs(iter_folder: Path) -> None:
    """
    Verify that required existing pipeline outputs are present.
    Raises FileNotFoundError with a clear message if anything is missing.
    """
    required = [
        iter_folder / "enriched_reduced.json",
    ]
    missing = [p for p in required if not p.exists()]
    if missing:
        paths = "\n  ".join(str(p) for p in missing)
        raise FileNotFoundError(
            f"Required pipeline output(s) missing:\n  {paths}\n"
            "Ensure the fabrication pipeline has completed for this iteration "
            "before running SEM analysis."
        )


def _check_inspection_inputs(iter_folder: Path) -> None:
    """
    Verify that inspection inputs (SEM images + observed features) exist.
    Raises FileNotFoundError with a clear message if anything is missing.
    """
    required = [
        iter_folder / "inspection" / "sem_top.png",
        iter_folder / "inspection" / "sem_oblique.png",
        iter_folder / "inspection" / "observed_features.json",
    ]
    missing = [p for p in required if not p.exists()]
    if missing:
        paths = "\n  ".join(str(p) for p in missing)
        raise FileNotFoundError(
            f"Inspection input(s) missing:\n  {paths}\n"
            "SEM images and observed_features.json must be placed in the "
            "inspection/ subfolder before running deviation analysis."
        )


# ======================================================
# PIPELINE
# ======================================================

def run_sem_analysis(iter_folder: Path) -> Path:
    """
    Execute the full SEM deviation analysis pipeline for one iteration.

    Args:
        iter_folder: Path to the iteration folder (e.g. Outputs/JobName/iter_000/)

    Returns:
        Path to the written deviation_report.json

    Raises:
        FileNotFoundError: if required inputs are missing
        ValueError: if any stage produces invalid output
    """
    iter_folder = iter_folder.resolve()

    if not iter_folder.is_dir():
        raise NotADirectoryError(
            f"Iteration folder not found: {iter_folder}"
        )

    print(f"[SEM] starting analysis for: {iter_folder}")

    # ── Preflight ────────────────────────────────────────────────────────────
    _check_pipeline_outputs(iter_folder)
    _check_inspection_inputs(iter_folder)

    # ── Step 1 + 2: Extract expected features from enriched_reduced.json ─────
    # extract_expected() reads enriched_reduced.json and writes
    # analysis/expected_features.json. Returns the path that was written.
    expected_features_path = extract_expected(iter_folder)

    # Read the written file back for the agent call
    with expected_features_path.open("r", encoding="utf-8") as fh:
        expected_features = json.load(fh)

    # ── Step 3: Load observed features ───────────────────────────────────────
    observed_features = load_observed(iter_folder)

    # ── Step 4: Confirm SEM images are readable (already checked in preflight)
    # Images are passed by path to the agent; no additional loading needed here.

    # ── Step 5: Run deviation agent ──────────────────────────────────────────
    deviation_report = run_agent(
        iter_folder=iter_folder,
        expected_features=expected_features,
        observed_features=observed_features,
    )

    # ── Step 6: Write deviation report ───────────────────────────────────────
    print("[SEM] writing deviation report")

    analysis_dir = iter_folder / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    report_path = analysis_dir / "deviation_report.json"
    with report_path.open("w", encoding="utf-8") as fh:
        json.dump(deviation_report, fh, indent=2)

    print(f"[SEM] deviation report written → {report_path}")
    _print_summary(deviation_report)

    return report_path


# ======================================================
# SUMMARY PRINTER
# ======================================================

def _print_summary(report: dict) -> None:
    """Print a brief human-readable summary of the deviation report."""
    sc = report.get("structure_check", {})
    failures = report.get("failures", [])
    feature_checks = report.get("feature_checks", [])

    match_str = "PASS" if sc.get("structure_match") else "FAIL"
    print(f"[SEM] structure check: {match_str}")
    print(f"[SEM] features checked: {len(feature_checks)}")

    if failures:
        print(f"[SEM] failures detected: {len(failures)}")
        for f in failures:
            print(
                f"[SEM]   [{f.get('failure_type', '?')}] "
                f"feature={f.get('feature_id', '?')} — "
                f"{f.get('description', '')}"
            )
    else:
        print("[SEM] no failures detected")


# ======================================================
# CLI ENTRY
# ======================================================

def main() -> None:
    if len(sys.argv) != 2:
        print(
            "Usage: python run_sem_analysis.py <iter_folder>\n"
            "Example: python run_sem_analysis.py Outputs/JobName/iter_000/"
        )
        sys.exit(1)

    iter_folder = Path(sys.argv[1])

    try:
        run_sem_analysis(iter_folder)
    except (FileNotFoundError, NotADirectoryError) as exc:
        print(f"[SEM] ERROR: {exc}")
        sys.exit(1)
    except ValueError as exc:
        print(f"[SEM] VALIDATION ERROR: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
