"""
Quick-start script for running the Named Object Agent from this directory.

Usage:
    python run_pipeline.py
    python run_pipeline.py "Your prompt here"
"""

import sys
from pathlib import Path

# Add src and parent directories to path
CURRENT_DIR = Path(__file__).parent
SRC_DIR = CURRENT_DIR / "src"
PARENT_DIR = CURRENT_DIR.parent

sys.path.insert(0, str(PARENT_DIR))
sys.path.insert(0, str(SRC_DIR))

from NamedObjectAgent import run_design


def main():
    # Get prompt from command line or file
    if len(sys.argv) > 1:
        prompt = " ".join(sys.argv[1:])
        category = "CLI"
    else:
        # Read from prompt.txt
        prompt_file = Path(__file__).parent / "prompt.txt"
        if not prompt_file.exists():
            print("ERROR: No prompt provided and prompt.txt not found")
            print("Usage: python run_pipeline.py \"Your prompt here\"")
            sys.exit(1)
        
        with open(prompt_file) as f:
            content = f.read().strip()
        
        # Parse category-prompt format
        lines = [l for l in content.split('\n') if l.strip() and not l.strip().startswith('#')]
        if not lines:
            print("ERROR: prompt.txt is empty")
            sys.exit(1)
        
        line = lines[0]
        if '-' in line:
            idx = line.index('-')
            category = line[:idx].strip()
            prompt = line[idx+1:].strip()
        else:
            category = "Default"
            prompt = line.strip()
    
    print("=" * 70)
    print("NAMED OBJECT AGENT V2")
    print("=" * 70)
    print(f"Category: {category}")
    print(f"Prompt: {prompt[:80]}...")
    print("=" * 70)
    
    try:
        result = run_design(prompt, category)
        
        print("\n" + "=" * 70)
        print("SUCCESS")
        print("=" * 70)
        print(f"Job name: {result['job_name']}")
        print(f"Objects defined: {result['num_objects']}")
        print(f"Total primitives: {result['num_primitives']}")
        print(f"Tokens used: {result['tokens']}")
        print(f"Output path: {result['output_path']}")
        
        if result.get('errors'):
            print(f"\nWarnings: {result['errors']}")
        
    except ValueError as e:
        print(f"\n[FATAL] v2 structural violation: {e}")
        print("The LLM produced v1 format output. Try re-running or adjust prompt.")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
