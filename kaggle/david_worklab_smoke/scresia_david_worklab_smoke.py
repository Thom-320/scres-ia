from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


GIT_URL = "https://github.com/Thom-320/scres-ia.git"
GIT_BRANCH = "codex/garrido-replication-experiments"
ROOT = Path("/kaggle/working/scres-ia")
RESULT = Path("/kaggle/working/david_worklab_smoke_result.json")


def run(cmd: list[str], *, cwd: Path | None = None) -> None:
    print("$", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=str(cwd) if cwd else None, text=True, capture_output=True)
    if proc.stdout:
        print(proc.stdout[-4000:], flush=True)
    if proc.returncode != 0:
        if proc.stderr:
            print(proc.stderr[-4000:], flush=True)
        raise RuntimeError(f"Command failed ({proc.returncode}): {cmd}")


def main() -> None:
    if not ROOT.exists():
        run(["git", "clone", "--depth", "1", "--branch", GIT_BRANCH, GIT_URL, str(ROOT)])
    else:
        print(f"Repo already exists: {ROOT}", flush=True)

    run([sys.executable, "-m", "pip", "install", "-q", "-r", str(ROOT / "requirements.txt")])

    sys.path.insert(0, str(ROOT))
    import einops  # noqa: F401
    from supply_chain.external_env_interface import make_dkana_thesis_faithful_env, make_track_b_env

    notebook_path = ROOT / "notebooks" / "david_track_a_track_b_colab_kaggle_playground.ipynb"
    notebook = json.loads(notebook_path.read_text())
    config_text = "".join(notebook["cells"][3]["source"])
    setup_text = "".join(notebook["cells"][4]["source"])
    assert 'REPO_SOURCE = "auto"' in config_text
    assert 'KAGGLE_REPO_PATH = "/kaggle/working/scres-ia"' in config_text
    assert "ROOT handle:" in setup_text

    env_a = make_dkana_thesis_faithful_env(max_steps=2, observation_version="v5")
    obs_a, _ = env_a.reset(seed=1)
    obs_a2, reward_a, terminated_a, truncated_a, _ = env_a.step(env_a.action_space.sample())
    env_a.close()

    env_b = make_track_b_env(max_steps=2, observation_version="v7", reward_mode="control_v1")
    obs_b, _ = env_b.reset(seed=1)
    obs_b2, reward_b, terminated_b, truncated_b, _ = env_b.step(env_b.action_space.sample())
    env_b.close()

    payload = {
        "status": "ok",
        "git_branch": GIT_BRANCH,
        "notebook": str(notebook_path),
        "repo_root": str(ROOT),
        "track_a": {
            "obs_shape": list(obs_a.shape),
            "next_obs_shape": list(obs_a2.shape),
            "action_space": str(env_a.action_space),
            "reward": float(reward_a),
            "done": bool(terminated_a or truncated_a),
        },
        "track_b": {
            "obs_shape": list(obs_b.shape),
            "next_obs_shape": list(obs_b2.shape),
            "action_space": str(env_b.action_space),
            "reward": float(reward_b),
            "done": bool(terminated_b or truncated_b),
        },
    }
    RESULT.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
