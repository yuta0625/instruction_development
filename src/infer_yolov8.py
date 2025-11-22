#YOLOv8で一括推論して、画像+txt+CSVを出すスクリプト
import yaml
from pathlib import Path
from typing import Any, Iterable
from ultralytics import YOLO
import csv, json
from .yolo_utils import yolo_results_to_dicts


def load_cfg(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def save_results_as_csv(results: Iterable[Any], out_dir: Path, filename: str = "predictions.csv") -> Path:

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    
    dets = yolo_results_to_dicts(results)

    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "image_path",
                "bbox_x1",
                "bbox_y1",
                "bbox_x2",
                "bbox_y2",
                "class_id",
                "class_name",
                "score",
            ]
        )

        for d in dets:
          writer.writerow(
            [
              d["image_path"],
              d["bbox"]["x1"],
              d["bbox"]["y1"],
              d["bbox"]["x2"],
              d["bbox"]["y2"],
              d["class_id"],
              d["class_name"],
              d["score"],
            ]
          )
    return out_path

def save_results_as_json(results: Iterable[Any], out_dir: Path, filename: str = "predictions.json") -> Path:
  out_dir.mkdir(parents=True, exist_ok=True)
  out_path = out_dir / filename
  
  dets = yolo_results_to_dicts(results)
  
  with out_path.open("w", encoding="utf-8") as f:
    json.dump(dets, f, ensure_ascii=False, indent=2)
  
  return out_path

def run_inference(cfg_path: str = "configs/yolov8_baseline.yaml", step: int | None = None) -> list:
    cfg = load_cfg(cfg_path)

    model = YOLO(cfg["model_name"])
    
    source = cfg["source_dir"]
    
    if step is not None:
      img_dir = Path(source)
      all_imgs = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
      #1/step
      source = all_imgs[::step]
    
    results = model.predict(
        source=source,
        imgsz=cfg["imgsz"],
        conf=cfg["conf_thres"],
        iou=cfg["iou_thres"],
        project=cfg["output_dir"],
        name=cfg.get("run_name", "exp"),
        save=cfg.get("save_img", True),
        save_txt=cfg.get("save_txt", True),
        verbose=True,
    )
    
    results = list(results)
    
    result_root = Path(cfg.get("result_dir", "runs/result"))
    result_dir = result_root / cfg.get("run_name", "exp")

    if cfg.get("save_csv", False):
        save_results_as_csv(results, result_dir)
        
    if cfg.get("save_json", False):
      save_results_as_json(results, result_dir)

    return results