"""Create a fully documented experiment run folder."""

from __future__ import annotations

import argparse
import datetime as dt
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run_text(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, cwd=ROOT, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"unavailable: {exc}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--command", required=True)
    args = parser.parse_args()

    run_dir = ROOT / "experiments" / "runs" / args.run_name
    for subdir in ("configs", "checkpoints", "logs", "predictions", "environment"):
        (run_dir / subdir).mkdir(parents=True, exist_ok=True)

    config_path = ROOT / args.config
    if config_path.is_file():
        shutil.copy2(config_path, run_dir / "configs" / config_path.name)
    if (ROOT / "requirements.txt").is_file():
        shutil.copy2(ROOT / "requirements.txt", run_dir / "environment" / "requirements.txt")
    if (ROOT / "environment.yml").is_file():
        shutil.copy2(ROOT / "environment.yml", run_dir / "environment" / "environment.yml")

    (run_dir / "logs" / "command.txt").write_text(args.command + "\n", encoding="utf-8")
    (run_dir / "logs" / "git_commit.txt").write_text(run_text(["git", "rev-parse", "HEAD"]) + "\n", encoding="utf-8")
    (run_dir / "logs" / "git_status.txt").write_text(run_text(["git", "status", "--short"]) + "\n", encoding="utf-8")
    (run_dir / "logs" / "pip_freeze.txt").write_text(run_text(["python", "-m", "pip", "freeze"]) + "\n", encoding="utf-8")

    template = (ROOT / "experiments" / "_templates" / "README.md").read_text(encoding="utf-8")
    readme = (
        template.replace("{{RUN_NAME}}", args.run_name)
        .replace("{{COMMAND}}", args.command)
        .replace("{{CONFIG}}", args.config)
        .replace("{{CREATED_AT}}", dt.datetime.now().isoformat(timespec="seconds"))
        .replace("{{GIT_COMMIT}}", run_text(["git", "rev-parse", "HEAD"]))
    )
    (run_dir / "README.md").write_text(readme, encoding="utf-8")
    print(run_dir)


if __name__ == "__main__":
    main()
