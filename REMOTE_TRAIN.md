# Remote GPU training (push → pull → train → push → pull)

Train on a CUDA GPU box (Colab / cloud / desktop with NVIDIA) instead of the Mac
CPU. The Mac's MPS GPU segfaults on YOLO-seg, so local training is CPU-only and
~10–40× slower — offload it here.

Everything needed is committed: `train_multi.py` + the exported `*_dataset/`
folders (each dataset ships its own image copies + `data.yaml`, so nothing else
is required). `labels.db` and the raw image pool are NOT needed to train from a
pre-built dataset.

## On the GPU machine (first time)
```bash
git clone https://github.com/zenblen/ZenblenImageTesting.git
cd ZenblenImageTesting
pip install ultralytics            # pulls torch (CUDA build) too
```

## Each run
```bash
cd ZenblenImageTesting
git pull                           # get the latest dataset + code

cd training
# --device 0 = first CUDA GPU. Modes: standard | spill | logo | chunk | unmixed | blended
python train_multi.py --mode blended --device 0

# best weights land here (git-allowed by .gitignore):
#   training/runs/blended-seg/blended-nano-v1/weights/best.pt
git add training/runs/blended-seg/*/weights/best.pt
git commit -m "Train blended-seg on GPU"
git push
```

`yolo11n-seg.pt` (the base model) auto-downloads on first run — the box just
needs internet.

## Back on the Mac
```bash
git pull
# deploy into the live pipeline (blended example):
cp training/runs/blended-seg/blended-nano-v1/weights/best.pt \
   active_pipeline/checkpoints/yolo_blended_seg.pt
```

## Notes
- Speed: 100 epochs ≈ ~10–15 min on a 4090/A100, ~30–50 min on a free Colab T4,
  vs ~4–5 hrs on the Mac CPU.
- Tune per run: `--epochs 100 --imgsz 640 --batch 16 --patience 30` (defaults).
  Bump `--batch` on a big GPU (e.g. 32/64) to go faster.
- To retrain after adding labels: re-export first
  (`python labeling/export_multi.py --mode <mode>`), commit the updated dataset,
  then pull + train on the GPU box. The `blended` (inverse) dataset is rebuilt
  with `experimentation/inverse_blend/build_inverse_dataset.py` (needs the Mac —
  it reads YOLO ROI + labels.db), then commit + pull + train.
```
