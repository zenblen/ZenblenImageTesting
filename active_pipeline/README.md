Active YOLO pipeline (blend/chunk + spill).

  cd active_pipeline
  python run.py --pipeline blend --image <img.jpg>
  python run.py --pipeline spill --image <img.jpg>

Weights (required):
  checkpoints/yolo_standard_seg.pt
  checkpoints/yolo_chunk_seg.pt
  checkpoints/yolo_spill_seg.pt

Deps: JetPack torch + ultralytics + opencv + numpy (see requirements.txt).
