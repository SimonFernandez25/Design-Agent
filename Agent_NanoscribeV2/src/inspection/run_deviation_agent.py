"""
run_deviation_agent.py - SEM Deviation Agent Interface

Calls an LLM agent to compare expected vs observed geometry from SEM images.

INPUTS:
    analysis/expected_features.json   - what we designed (from extract_expected_features)
    inspection/observed_features.json - what segmentation detected
    inspection/sem_top.png            - top-down SEM image
    inspection/sem_oblique.png        - oblique SEM image

OUTPUT:
    analysis/deviation_report.json    - structured deviation classification

STRICT AGENT CONSTRAINTS (enforced by system prompt + output schema):
    - The agent may ONLY classify deviations
    - The agent may NOT reconstruct CAD
    - The agent may NOT modify geometry
    - The agent may NOT infer printer parameters

Allowed failure_type values (fixed enum, agent cannot add new ones):
    missing_feature        - expected feature absent from SEM image
    merged_features        - two separate features appear fused
    collapsed_structure    - feature has lost structural height/shape
    filled_hole            - expected void/hole appears filled
    global_collapse        - entire structure shows failure
    unrecognizable_structure - SEM image too degraded to classify
"""

import base64
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI


# ======================================================
# CONFIG
# ======================================================

def _find_api_key(start_dir: Path, max_levels: int = 6) -> str:
    """
    Walk up directory tree looking for Docs/API.txt (same convention as
    the rest of the Design-Agent codebase).
    """
    current = start_dir
    for _ in range(max_levels):
        candidate = current / "Docs" / "API.txt"
        if candidate.exists():
            return candidate.read_text(encoding="utf-8").strip()
        parent = current.parent
        if parent == current:
            break
        current = parent

    # Fallback to environment variable
    key = os.environ.get("OPENAI_API_KEY", "")
    if key:
        return key

    raise ValueError(
        "API key not found. Looked for Docs/API.txt up the directory tree "
        "and OPENAI_API_KEY environment variable."
    )


_SCRIPT_DIR = Path(__file__).parent.parent.parent.resolve()
_API_KEY    = _find_api_key(_SCRIPT_DIR)

_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=_API_KEY,
)

MODEL = "openai/gpt-4o"


# ======================================================
# OUTPUT SCHEMA
# ======================================================

DEVIATION_REPORT_SCHEMA = {
    "type": "object",
    "properties": {
        "structure_check": {
            "type": "object",
            "properties": {
                "expected_structure": {
                    "type": "string",
                    "description": "Plain-language description of expected geometry"
                },
                "observed_structure": {
                    "type": "string",
                    "description": "Plain-language description of observed SEM geometry"
                },
                "structure_match": {
                    "type": "boolean",
                    "description": "True if overall observed structure matches expected"
                }
            },
            "required": ["expected_structure", "observed_structure", "structure_match"],
            "additionalProperties": False
        },
        "feature_checks": {
            "type": "array",
            "description": "Per-feature deviation classification",
            "items": {
                "type": "object",
                "properties": {
                    "feature_id": {
                        "type": "string",
                        "description": "Identifier matching object_id or hole_id from expected_features"
                    },
                    "expected_type": {
                        "type": "string",
                        "description": "Primitive type as listed in expected_features (e.g. 'cylinder', 'box')"
                    },
                    "observed_state": {
                        "type": "string",
                        "description": "Short description of what is actually observed in the SEM image"
                    },
                    "deviation_type": {
                        "type": "string",
                        "enum": [
                            "none",
                            "missing_feature",
                            "merged_features",
                            "collapsed_structure",
                            "filled_hole",
                            "global_collapse",
                            "unrecognizable_structure"
                        ],
                        "description": "Deviation class. Use 'none' if feature matches expectation."
                    },
                    "confidence": {
                        "type": "string",
                        "enum": ["high", "medium", "low"],
                        "description": "Agent confidence in this classification"
                    }
                },
                "required": [
                    "feature_id", "expected_type",
                    "observed_state", "deviation_type", "confidence"
                ],
                "additionalProperties": False
            }
        },
        "failures": {
            "type": "array",
            "description": "List of confirmed fabrication failures (only when deviation_type != 'none')",
            "items": {
                "type": "object",
                "properties": {
                    "failure_type": {
                        "type": "string",
                        "enum": [
                            "missing_feature",
                            "merged_features",
                            "collapsed_structure",
                            "filled_hole",
                            "global_collapse",
                            "unrecognizable_structure"
                        ]
                    },
                    "feature_id": {
                        "type": "string",
                        "description": "Feature this failure refers to"
                    },
                    "description": {
                        "type": "string",
                        "description": "One-sentence description of the failure"
                    }
                },
                "required": ["failure_type", "feature_id", "description"],
                "additionalProperties": False
            }
        }
    },
    "required": ["structure_check", "feature_checks", "failures"],
    "additionalProperties": False
}


# ======================================================
# SYSTEM PROMPT
# ======================================================

DEVIATION_AGENT_SYSTEM_PROMPT = """You are a Scanning Electron Microscopy (SEM) deviation analysis agent.

Your ONLY job is to compare the expected geometry (from design files) with what is actually observed in SEM images, and classify any deviations.

ABSOLUTE RULES — VIOLATION CAUSES IMMEDIATE FAILURE:
1. You may ONLY classify deviations. Nothing else.
2. You may NOT reconstruct CAD geometry.
3. You may NOT modify any geometry values.
4. You may NOT infer or suggest printer parameter changes.
5. You may NOT add new failure categories beyond the fixed allowed list.
6. You may NOT speculate on causes of failure — only classify what is observed.

DEVIATION CLASSIFICATION WORKFLOW:
1. Read expected_features.json to understand what geometry should be present.
2. Read observed_features.json to understand the segmentation statistics.
3. Study the SEM images (top and oblique views).
4. For each expected object/hole: determine if it matches observation, or classify the deviation.
5. Compile failures list from non-'none' deviations.

ALLOWED DEVIATION TYPES (exact strings only):
- "none"                     : Feature matches expectation, no deviation
- "missing_feature"          : Expected feature is completely absent from SEM image
- "merged_features"          : Two or more separate features appear fused into one
- "collapsed_structure"      : Feature lost structural height or form (sagged, bent, or spread)
- "filled_hole"              : An expected void or hole appears filled or blocked
- "global_collapse"          : Entire structure shows catastrophic failure
- "unrecognizable_structure" : SEM image is too degraded or noisy to classify

CONFIDENCE LEVELS:
- "high"   : Clear visual evidence in both SEM views
- "medium" : Visible but ambiguous (consider using oblique view)
- "low"    : Inferred from statistics only; not clearly visible

OUTPUT RULES:
- structure_check.expected_structure: Describe the design geometry in plain language (count, type, arrangement).
- structure_check.observed_structure: Describe what you actually see in the SEM images.
- structure_check.structure_match: true ONLY if the overall structure matches expected.
- feature_checks: One entry per object/hole from expected_features. Always include ALL expected features.
- failures: Include only features where deviation_type != 'none'. May be empty list if all features match.

You output valid JSON matching the required schema. No prose. No explanations outside the schema."""


# ======================================================
# IMAGE UTILITIES
# ======================================================

def _encode_image_base64(image_path: Path) -> str:
    """Encode image file to base64 string for multimodal API call."""
    with image_path.open("rb") as fh:
        return base64.b64encode(fh.read()).decode("utf-8")


def _image_message_block(image_path: Path, label: str) -> dict:
    """
    Build a content block for the multimodal message.
    Includes a text label followed by the base64 image.
    """
    b64 = _encode_image_base64(image_path)
    suffix = image_path.suffix.lower().lstrip(".")
    media_map = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "tiff": "tiff", "tif": "tiff"}
    media_type = f"image/{media_map.get(suffix, 'png')}"

    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:{media_type};base64,{b64}",
            "detail": "high"
        }
    }


# ======================================================
# VALIDATION
# ======================================================

ALLOWED_FAILURE_TYPES = {
    "missing_feature",
    "merged_features",
    "collapsed_structure",
    "filled_hole",
    "global_collapse",
    "unrecognizable_structure",
}

ALLOWED_DEVIATION_TYPES = ALLOWED_FAILURE_TYPES | {"none"}


def _validate_report(report: Dict[str, Any]) -> List[str]:
    """
    Post-LLM validation of the deviation report.
    Returns list of violation messages (empty = valid).
    """
    violations: List[str] = []

    # Check top-level keys
    for key in ("structure_check", "feature_checks", "failures"):
        if key not in report:
            violations.append(f"Missing required top-level key: '{key}'")

    # Validate feature_checks deviation types
    for fc in report.get("feature_checks", []):
        dtype = fc.get("deviation_type", "")
        if dtype not in ALLOWED_DEVIATION_TYPES:
            violations.append(
                f"feature_checks[{fc.get('feature_id', '?')}]: "
                f"invalid deviation_type '{dtype}'"
            )

    # Validate failures failure types
    for fail in report.get("failures", []):
        ftype = fail.get("failure_type", "")
        if ftype not in ALLOWED_FAILURE_TYPES:
            violations.append(
                f"failures[{fail.get('feature_id', '?')}]: "
                f"invalid failure_type '{ftype}'"
            )

    return violations


# ======================================================
# AGENT CALL
# ======================================================

def run_deviation_agent(
    expected_features: Dict[str, Any],
    observed_features: Dict[str, Any],
    sem_top_path: Path,
    sem_oblique_path: Path,
) -> Dict[str, Any]:
    """
    Call the LLM deviation agent and return the structured deviation report.

    Args:
        expected_features:  Dict from expected_features.json
        observed_features:  Dict from observed_features.json
        sem_top_path:       Path to sem_top.png
        sem_oblique_path:   Path to sem_oblique.png

    Returns:
        deviation_report dict conforming to the deviation_report.json schema

    Raises:
        ValueError: if the LLM output fails schema validation
        FileNotFoundError: if SEM image paths are missing
    """
    print("[SEM] running deviation agent")

    # Validate SEM image paths
    for img_path in (sem_top_path, sem_oblique_path):
        if not img_path.exists():
            raise FileNotFoundError(
                f"SEM image not found: {img_path}\n"
                "Both sem_top.png and sem_oblique.png must exist in the "
                "inspection/ folder before running deviation analysis."
            )

    # Build user message content (text + images)
    user_content = [
        {
            "type": "text",
            "text": (
                "## Expected Features (from design file)\n"
                f"```json\n{json.dumps(expected_features, indent=2)}\n```\n\n"
                "## Observed Features (from segmentation)\n"
                f"```json\n{json.dumps(observed_features, indent=2)}\n```\n\n"
                "## SEM Images\n"
                "Top-down SEM view:"
            )
        },
        _image_message_block(sem_top_path, "SEM top-down view"),
        {
            "type": "text",
            "text": "Oblique SEM view:"
        },
        _image_message_block(sem_oblique_path, "SEM oblique view"),
        {
            "type": "text",
            "text": (
                "\nCompare the expected features against the observed features "
                "and SEM images. Classify any deviations per the schema. "
                "Output only valid JSON."
            )
        }
    ]

    response = _client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": DEVIATION_AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "deviation_report",
                "strict": False,
                "schema": DEVIATION_REPORT_SCHEMA,
            }
        },
        temperature=0.0,
    )

    raw_content = response.choices[0].message.content
    report = json.loads(raw_content)

    # Post-LLM validation: enforce failure type enum strictly
    violations = _validate_report(report)
    if violations:
        formatted = "\n  ".join(violations)
        raise ValueError(
            f"Deviation agent output failed validation "
            f"({len(violations)} violation(s)):\n  {formatted}\n"
            f"Raw output:\n{raw_content}"
        )

    return report


# ======================================================
# FILE-LEVEL RUNNER
# ======================================================

def run(
    iter_folder: Path,
    expected_features: Optional[Dict[str, Any]] = None,
    observed_features: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Run the deviation agent for one iteration folder.

    Reads from:
        analysis/expected_features.json
        inspection/observed_features.json
        inspection/sem_top.png
        inspection/sem_oblique.png

    Returns: deviation_report dict (caller is responsible for writing to disk)
    """
    if expected_features is None:
        ef_path = iter_folder / "analysis" / "expected_features.json"
        with ef_path.open("r", encoding="utf-8") as fh:
            expected_features = json.load(fh)

    if observed_features is None:
        of_path = iter_folder / "inspection" / "observed_features.json"
        with of_path.open("r", encoding="utf-8") as fh:
            observed_features = json.load(fh)

    sem_top_path     = iter_folder / "inspection" / "sem_top.png"
    sem_oblique_path = iter_folder / "inspection" / "sem_oblique.png"

    return run_deviation_agent(
        expected_features=expected_features,
        observed_features=observed_features,
        sem_top_path=sem_top_path,
        sem_oblique_path=sem_oblique_path,
    )


# ======================================================
# CLI ENTRY (for standalone testing)
# ======================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python run_deviation_agent.py <iter_folder>")
        sys.exit(1)

    result = run(Path(sys.argv[1]))
    print(json.dumps(result, indent=2))
