# Nanoscribe Design Agent V2 (Simplified)

This folder contains a self-contained, simplified version of the **Named Object Architecture (v2)** for Nanoscribe two-photon lithography.

## Architecture

```
Natural Language Prompt
         |
         v
+------------------------+
|  NamedObjectAgent      |  <-- LLM generates objects + assembly JSON (V2)
+------------------------+
         |
         v
+------------------------+
|  v2_structural_gate    |  <-- BLOCKS legacy v1 output (unit_cell)
+------------------------+
         |
         v
+------------------------+
|  Reduction Engine      |  <-- Flattens to absolute primitives
+------------------------+
         |
         v
+------------------------+
|  endpoint_generator_v2 |  <-- Consumes reduced.json
+------------------------+
         |
         v
+------------------------+
|  GWL Serializer        |  <-- Nanoscribe format
+------------------------+
         |
         v
     .gwl files ready for NanoWrite
```

## Quick Start

### 1. Setup

```bash
cd Agent_NanoscribeV1
pip install -r requirements.txt
# Ensure ../Docs/API.txt exists with your OpenAI API key
```

### 2. Run Pipeline

```bash
# Edit prompt.txt, then run:
python run_pipeline.py

# Or pass prompt string directly:
python run_pipeline.py "Create a 10x10 array of 5um cylinders"
```

### 3. Redesign (Human-in-the-Loop)

To refine an existing design:

```bash
# From Agent_NanoscribeV1 folder:
python redesign/Redesign_Agent_v2.py "MyProject" Outputs/MyProject_.../design.json PrintParameters.txt "Make the cylinders taller"
```

## File Structure

```
Agent_NanoscribeV1/
    NamedObjectAgent.py     # Main Agent V2
    reduction_engine.py     # Flattens object hierarchy
    endpoint_generator_v2.py# Generates scan paths
    render_generator_v2.py  # Visualizes output
    gwl_serializer.py       # Outputs .gwl files
    schemas/                # V2 JSON Schemas
    redesign/               # Human-in-the-loop agents
        Redesign_Agent_v2.py
        edit_suggestion_agent_v2.py
        edit_executor_v2.py
        edit_schema_v2.py
```

## V2 Key Concepts

1. **Named Objects**: Define reusable geometry (e.g., "nailhead", "pillar").
2. **Assembly**: Define how objects are arranged (grid, mapping).
3. **No Unit Cells**: The concept of "unit_cell" is replaced by flexible named objects.
