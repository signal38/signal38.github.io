"""Shared helpers for Colab notebooks in this repo."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable

REPO_URL = "https://github.com/signal38/signal38.github.io.git"
DEFAULT_REPO_DIR = Path("/content/signal38.github.io")
NOTEBOOK_DEPS_SENTINEL_VERSION = "v1"


def bootstrap_colab_repo(
    repo_dir: str | Path = DEFAULT_REPO_DIR,
    repo_url: str = REPO_URL,
) -> Path:
    """Clone the repo into the Colab workspace if needed."""
    repo_path = Path(repo_dir)
    if not repo_path.exists():
        repo_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", repo_url, str(repo_path)], check=True)

    if str(repo_path) not in sys.path:
        sys.path.insert(0, str(repo_path))

    os.chdir(repo_path)
    return repo_path


def get_repo_paths(repo_root: str | Path = DEFAULT_REPO_DIR) -> dict[str, Path]:
    """Return canonical paths used across notebooks."""
    repo_path = Path(repo_root)
    return {
        "repo_root": repo_path,
        "notebooks_dir": repo_path / "notebooks",
        "data_root": repo_path / "data",
        "clusters": repo_path / "data" / "clusters" / "all_clusters.jsonl",
        "labeled": repo_path / "data" / "labeled" / "all_labeled.jsonl",
        "training_dir": repo_path / "data" / "training",
        "outputs_dir": repo_path / "data" / "outputs",
        "models_dir": repo_path / "models",
        "adapter_dir": repo_path / "models" / "lfm2-nk-risk" / "lora_adapter",
        "xgb_dir": repo_path / "models" / "xgb",
    }


def prepare_notebook(
    repo_root: str | Path = DEFAULT_REPO_DIR,
    *,
    pull_latest: bool = False,
) -> tuple[Path, dict[str, Path]]:
    """Bootstrap the repo and optionally pull latest main."""
    repo_path = bootstrap_colab_repo(repo_root)

    if pull_latest:
        try:
            from google.colab import userdata
            token = userdata.get("GITHUB_TOKEN")
        except Exception:
            token = None

        if token:
            authed_url = f"https://x-access-token:{token}@github.com/signal38/signal38.github.io.git"
            subprocess.run(["git", "remote", "set-url", "origin", authed_url], check=True, cwd=repo_path)

        try:
            subprocess.run(["git", "pull", "origin", "main"], check=True, cwd=repo_path)
        except subprocess.CalledProcessError:
            print("Warning: git pull failed — continuing with local repo state.")

    return repo_path, get_repo_paths(repo_path)


def ensure_notebook_requirements(
    notebook_name: str,
    *,
    requirements_path: str | Path = "../requirements.txt",
) -> None:
    """Install notebook dependencies once per session and restart runtime.

    The runtime restart ensures bitsandbytes loads against the correct CUDA
    binaries. Without this, quantized inference silently fails on Colab.
    """
    sentinel = Path(f"/tmp/signal38_{notebook_name}_deps_{NOTEBOOK_DEPS_SENTINEL_VERSION}")
    if sentinel.exists():
        print(f"Dependencies ready for {notebook_name}.")
        return

    subprocess.run(
        [
            sys.executable, "-m", "pip", "install", "-q", "-U",
            "--upgrade-strategy", "only-if-needed",
            "-r", str(requirements_path),
        ],
        check=True,
    )
    sentinel.write_text("ok")
    print("Dependencies updated. Restarting runtime for clean CUDA bindings...")
    os.kill(os.getpid(), 9)


def require_local_adapter(repo_root: str | Path = DEFAULT_REPO_DIR) -> Path:
    """Return the LoRA adapter path or raise with a clear notebook dependency message."""
    adapter_dir = get_repo_paths(repo_root)["adapter_dir"]
    if adapter_dir.exists():
        return adapter_dir
    raise FileNotFoundError(
        f"LoRA adapter not found at {adapter_dir}.\n"
        "Run notebook 02_finetune.ipynb and publish the adapter first, "
        "then pull latest main before running this notebook."
    )


def publish_artifacts(
    paths: Iterable[str | Path],
    message: str,
    repo_dir: str | Path = DEFAULT_REPO_DIR,
    dry_run: bool = False,
) -> bool:
    """Commit and push generated artifacts from Colab back to GitHub.

    Requires a Colab secret named ``GITHUB_TOKEN`` with write access to the
    signal38 org. Add it via the key icon in the Colab left sidebar.

    Returns True when a commit was created and pushed.
    """
    try:
        from google.colab import userdata
    except ImportError as exc:
        raise RuntimeError("publish_artifacts only works from Google Colab.") from exc

    token = userdata.get("GITHUB_TOKEN")
    if not token:
        raise RuntimeError(
            "Missing Colab secret GITHUB_TOKEN. "
            "Add it via the key icon in the Colab left sidebar."
        )

    repo_path = Path(repo_dir)
    rel_paths = [str(Path(p)) for p in paths]

    missing = [p for p in rel_paths if not (repo_path / p).exists()]
    if missing:
        raise FileNotFoundError(
            "Cannot publish — files not found: " + ", ".join(missing)
        )

    repo_url = f"https://x-access-token:{token}@github.com/signal38/signal38.github.io.git"
    stash_name = "colab-artifacts-publish"

    subprocess.run(["git", "config", "user.email", "colab-bot@signal38"], check=True, cwd=repo_path)
    subprocess.run(["git", "config", "user.name", "Colab Bot"], check=True, cwd=repo_path)
    subprocess.run(["git", "remote", "set-url", "origin", repo_url], check=True, cwd=repo_path)

    stash_push = subprocess.run(
        ["git", "stash", "push", "--include-untracked", "-m", stash_name, "--", *rel_paths],
        check=True, cwd=repo_path, capture_output=True, text=True,
    )
    # git stash writes its status message to stderr, not stdout
    stash_output = stash_push.stdout + stash_push.stderr
    stashed = "No local changes to save" not in stash_output

    try:
        subprocess.run(["git", "pull", "--rebase", "origin", "main"], check=True, cwd=repo_path)
    except subprocess.CalledProcessError as exc:
        if stashed:
            subprocess.run(["git", "stash", "pop"], cwd=repo_path)
        raise RuntimeError(
            "git pull --rebase failed before publishing. "
            "Check for remote conflicts and retry."
        ) from exc

    if stashed:
        pop = subprocess.run(["git", "stash", "pop"], cwd=repo_path, capture_output=True, text=True)
        if pop.returncode != 0:
            raise RuntimeError(
                "git stash pop failed after rebase — possible merge conflict.\n"
                + pop.stderr
            )

    subprocess.run(["git", "add", "--", *rel_paths], check=True, cwd=repo_path)

    # git diff --cached misses brand-new files with no HEAD entry; use status instead
    status = subprocess.run(
        ["git", "status", "--porcelain", "--", *rel_paths],
        cwd=repo_path, capture_output=True, text=True, check=True,
    )
    staged = [l for l in status.stdout.splitlines() if l and l[0] not in (" ", "?")]
    if not staged:
        print("No artifact changes to commit.")
        return False

    if dry_run:
        print(f"[dry_run] Would commit: {', '.join(rel_paths)}")
        print(f"[dry_run] Message: {message}")
        return False

    subprocess.run(["git", "commit", "-m", message], check=True, cwd=repo_path)

    push = subprocess.run(
        ["git", "push", "origin", "main"],
        cwd=repo_path, capture_output=True, text=True,
    )
    if push.returncode != 0:
        raise RuntimeError(
            f"git push failed (exit {push.returncode}):\n"
            + (push.stderr or push.stdout)
        )
    print(f"Pushed: {', '.join(rel_paths)}")
    return True
