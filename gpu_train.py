#!/usr/bin/env python
"""One-shot remote GPU training: pull -> train -> push weights back.

Cross-platform (Windows / Linux / Mac) — no bash needed. Run from anywhere in
the repo:

    python gpu_train.py                        # trains 'blended' on GPU 0
    python gpu_train.py chunk                  # any mode
    python gpu_train.py blended 0              # mode + device (0 = first CUDA GPU, or 'cpu')
    python gpu_train.py blended 0 --no-amp     # extra args pass straight through to train_multi.py

Modes: standard | spill | logo | chunk | unmixed | blended
Then on the Mac:  git pull  and copy the printed best.pt into active_pipeline/checkpoints/
"""
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent


def run(cmd, cwd=REPO, check=True):
    print(f">> {' '.join(str(c) for c in cmd)}", flush=True)
    return subprocess.run(cmd, cwd=str(cwd), check=check)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "blended"
    device = sys.argv[2] if len(sys.argv) > 2 else "0"
    extra = sys.argv[3:]                       # e.g. --no-amp, passed straight through
    project_dir = REPO / "training" / "runs" / f"{mode}-seg"

    print(">> [1/3] pull latest dataset + code")
    run(["git", "pull", "--ff-only"])

    print(f">> [2/3] train {mode} on device {device}" + (f" {' '.join(extra)}" if extra else ""))
    run([sys.executable, "train_multi.py", "--mode", mode, "--device", device, *extra],
        cwd=REPO / "training")

    # Find the newest best.pt for this mode — auto-incremented run names
    # (blended-nano-v1, -v2, ...) mean the exact run dir isn't known in advance,
    # e.g. after a crashed/retried attempt.
    candidates = sorted(project_dir.glob("*/weights/best.pt"),
                        key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        sys.exit(f"!! no weights found under {project_dir}\n"
                 f"!! (did training finish? check training/runs/{mode}-seg/)")
    best = candidates[0]

    print(">> [3/3] push weights back")
    run(["git", "add", str(best)])
    # commit returns nonzero if there is nothing to commit — that's fine.
    if run(["git", "commit", "-m", f"Train {mode}-seg on GPU (device {device})"],
           check=False).returncode != 0:
        print("nothing to commit (weights unchanged)")
        return
    run(["git", "push"])

    rel = best.relative_to(REPO).as_posix()
    print(f">> done. On the Mac:  git pull  then copy:\n"
          f"   {rel}  ->  active_pipeline/checkpoints/yolo_{mode}_seg.pt")


if __name__ == "__main__":
    main()
