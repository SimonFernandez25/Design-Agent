# Changelog

## [V2 Simplified] - 2026-02-05

### Changed
- **Architecture Upgrade**: Fully transitioned from V1 (Unit Cell Agent) to V2 (Named Object Agent).
- **Core Pipeline**:
  - Replaced `unit_cell_prototype.py` with `NamedObjectAgent.py`.
  - Added `reduction_engine.py` to handle object hierarchy flattening.
  - Updated `endpoint_generator.py` to `endpoint_generator_v2.py` for V2 compatibility.
  - Updated `render_generator.py` to `render_generator_v2.py` for object-aware rendering.
  - Added `gwl_serializer.py` as a standalone module.
- **Redesign System**:
  - Updated `redesign/` folder with V2 Human-in-the-Loop agents.
  - Renamed `edit_suggestion_agent_v2_final.py` to `edit_suggestion_agent_v2.py`.
  - `Redesign_Agent_v2.py` replaces legacy redesign scripts.

### Removed
- Legacy V1 scripts and "Unit Cell" specific logic.
- References to `global_info` and `unit_cell` fields in JSON schemas.
