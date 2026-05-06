import argparse
import subprocess
import sys
from pathlib import Path


COMMANDS = {
    "env": [sys.executable, "run_pipeline_debug.py", "--dry-run"],
    "fake": [sys.executable, "generate_fake_results_presentation.py"],
    "fake-fast": [sys.executable, "generate_fake_results_fast.py"],
    "fake-large": [sys.executable, "generate_fake_results_large.py"],
    "debug": [sys.executable, "run_pipeline_debug.py", "--preview"],
    "debug-skip-train": [sys.executable, "run_pipeline_debug.py", "--skip-train", "--preview"],
    "audit-only": [sys.executable, "run_pipeline_debug.py", "--skip-preprocess", "--skip-features", "--skip-train", "--skip-ranking", "--preview"],
    "viewer": [sys.executable, "-m", "streamlit", "run", "src/ui/viewer_app.py"],
    "api": [sys.executable, "-m", "uvicorn", "src.api.api_server:app", "--reload"],
}


def run_command(name):
    cmd = COMMANDS[name]
    print("=" * 96)
    print("running:", " ".join(cmd))
    print("=" * 96)
    completed = subprocess.run(cmd, cwd=Path.cwd())
    return completed.returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=COMMANDS.keys())
    args = parser.parse_args()
    raise SystemExit(run_command(args.command))


if __name__ == "__main__":
    main()
