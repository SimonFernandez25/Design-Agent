"""
test_manufacturing_agent.py - Unit tests for the manufacturing agent

Tests:
1. Metric extraction for box and cylinder primitives
2. Risk score bounds (must be in [0, 1])
3. Local parameter clamp bounds
4. Full enrich_primitives() contract
"""

import sys
import math
from pathlib import Path

# Ensure Agent_NanoscribeV2 is on path
sys.path.insert(0, str(Path(__file__).parent))

import manufacturing_agent as ma

PASS = "[PASS]"
FAIL = "[FAIL]"


def _assert(condition: bool, label: str, detail: str = ""):
    if condition:
        print(f"{PASS} {label}")
    else:
        print(f"{FAIL} {label}" + (f" -- {detail}" if detail else ""))
    return condition


def make_reduced_data():
    """Build a minimal reduced.json dict with 3 representative primitives."""
    return {
        "job_name": "test_mfg_agent",
        "primitives": [
            {
                "type": "cylinder",
                "center": [0.0, 0.0, 5.0],
                "dimensions": {"diameter_um": 2.0, "height_um": 10.0},
                "description": "thin tall cylinder"
            },
            {
                "type": "box",
                "center": [20.0, 0.0, 3.0],
                "dimensions": {"x_um": 20.0, "y_um": 20.0, "z_um": 6.0},
                "description": "fat squat box"
            },
            {
                "type": "box",
                "center": [40.0, 0.0, 15.0],
                "dimensions": {"x_um": 1.0, "y_um": 1.0, "z_um": 30.0},
                "description": "tall slender box"
            },
        ],
        "metadata": {"num_primitives": 3}
    }


def run_tests():
    print("=" * 60)
    print("Manufacturing Agent Unit Tests")
    print("=" * 60)

    all_pass = True

    # Test 1: enrich_primitives runs without error
    reduced = make_reduced_data()
    enriched = ma.enrich_primitives(reduced, output_dir=None)
    prims = enriched["primitives"]
    ok = _assert(len(prims) == 3, "enrich returns 3 primitives")
    all_pass = all_pass and ok

    for idx, p in enumerate(prims):
        label = f"primitive[{idx}] ({p.get('description', p.get('type'))})"

        # Test: metrics present
        ok = _assert("metrics" in p, f"{label} has metrics")
        all_pass = all_pass and ok

        # Test: risk present and in [0, 1]
        ok = _assert("risk" in p, f"{label} has risk")
        all_pass = all_pass and ok
        if "risk" in p:
            r = p["risk"]
            ok = _assert(0.0 <= r <= 1.0, f"{label} risk in [0,1]", f"got {r}")
            all_pass = all_pass and ok

        # Test: risk_factors present
        ok = _assert("risk_factors" in p, f"{label} has risk_factors")
        all_pass = all_pass and ok

        # Test: local_parameters present
        ok = _assert("local_parameters" in p, f"{label} has local_parameters")
        all_pass = all_pass and ok

        if "local_parameters" in p:
            lp = p["local_parameters"]

            h = lp["hatch_spacing_um"]
            ok = _assert(
                ma.HATCH_MIN <= h <= ma.HATCH_MAX,
                f"{label} hatch in [{ma.HATCH_MIN}, {ma.HATCH_MAX}]",
                f"got {h}"
            )
            all_pass = all_pass and ok

            s = lp["slice_spacing_um"]
            ok = _assert(
                ma.SLICE_MIN <= s <= ma.SLICE_MAX,
                f"{label} slice in [{ma.SLICE_MIN}, {ma.SLICE_MAX}]",
                f"got {s}"
            )
            all_pass = all_pass and ok

            c = lp["solid_contour_count"]
            ok = _assert(
                ma.CONTOURS_MIN <= c <= ma.CONTOURS_MAX,
                f"{label} contours in [{ma.CONTOURS_MIN}, {ma.CONTOURS_MAX}]",
                f"got {c}"
            )
            all_pass = all_pass and ok

    # Test: cylinder has radius in metrics
    cyl = prims[0]
    ok = _assert(
        "radius_um" in cyl.get("metrics", {}),
        "cylinder metrics contains radius_um"
    )
    all_pass = all_pass and ok

    # Test: slender box has higher risk than fat box
    risk_cyl  = prims[0]["risk"]   # thin tall cylinder -> high risk
    risk_fat  = prims[1]["risk"]   # fat squat box -> lower risk
    risk_tall = prims[2]["risk"]   # tall slender box -> highest risk
    print(f"\nRisk values: thin_cyl={risk_cyl:.4f}  fat_box={risk_fat:.4f}  tall_box={risk_tall:.4f}")
    ok = _assert(
        risk_tall > risk_fat,
        "tall slender box has higher risk than fat box"
    )
    all_pass = all_pass and ok

    # Test: high-risk primitive gets tighter hatch/slice
    lp_fat  = prims[1]["local_parameters"]
    lp_tall = prims[2]["local_parameters"]
    ok = _assert(
        lp_tall["hatch_spacing_um"] <= lp_fat["hatch_spacing_um"],
        "tall box gets denser hatch than fat box"
    )
    all_pass = all_pass and ok

    # Test: eps guard for degenerate zero-dimension primitive
    degenerate = {
        "type": "box",
        "center": [0, 0, 0],
        "dimensions": {"x_um": 0.0, "y_um": 0.0, "z_um": 0.0},
    }
    try:
        metrics_d = ma.compute_metrics(degenerate)
        risk_d, _ = ma.compute_risk(metrics_d)
        ok = _assert(0.0 <= risk_d <= 1.0, "degenerate primitive does not crash, risk in [0,1]")
    except Exception as ex:
        ok = _assert(False, "degenerate primitive does not crash", str(ex))
    all_pass = all_pass and ok

    print("\n" + "=" * 60)
    if all_pass:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED - see above")
    print("=" * 60)
    return all_pass


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
