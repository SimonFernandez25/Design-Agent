"""
SEM Deviation Analysis Module

Post-fabrication analysis branch that reads pipeline outputs and produces
deviation reports. This module never modifies existing pipeline files.

Components:
    extract_expected_features  - Parses enriched_reduced.json → expected_features.json
    load_observed_features     - Validates and loads inspection/observed_features.json
    run_deviation_agent        - Calls LLM agent to classify deviations
    run_sem_analysis           - Pipeline runner (top-level entry point)
"""
