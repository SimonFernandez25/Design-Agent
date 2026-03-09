"""
run_ui.py - Launcher for the Nanoscribe Design Agent UI

Starts the Streamlit app from the project root directory so all
relative imports and file paths resolve correctly.

Usage:
    py run_ui.py
    py run_ui.py --port 8502
    py run_ui.py --browser.serverAddress localhost
"""

import subprocess
import sys
from pathlib import Path

UI_SCRIPT = Path(__file__).parent / "src" / "ui" / "supervisor_ui.py"


def main():
    # Pass any extra CLI args straight through to streamlit
    extra_args = sys.argv[1:]

    cmd = [
        sys.executable, "-m", "streamlit", "run",
        str(UI_SCRIPT),
        "--server.headless", "false",
        *extra_args,
    ]

    print("=" * 60)
    print("Nanoscribe Design Agent UI")
    print("=" * 60)
    print(f"Launching: {UI_SCRIPT.name}")
    print("Open your browser at http://localhost:8501")
    print("Press Ctrl+C to stop.")
    print("=" * 60)

    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nUI stopped.")
    except subprocess.CalledProcessError as exc:
        print(f"\nError starting UI: {exc}")
        print("Make sure streamlit is installed:  pip install streamlit")
        sys.exit(1)


if __name__ == "__main__":
    main()
