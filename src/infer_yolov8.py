#YOLOv8で一括推論して、画像+txt+CSVを出すスクリプト
import yaml
from pathlib import Path
from typing import Any, List
from ultralytics import YOLO

def load_cfg(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def save_results_as_csv(results: List[Any], out_dir: Path, filename: str = "predictions.csv") -> Path:
    import csv

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename

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

        for r in results:
            img_path = r.path
            boxes = getattr(r, "boxes", None)
            if boxes is None:
                continue

            xyxy = boxes.xyxy.cpu().numpy()
            conf = boxes.conf.cpu().numpy()
            cls = boxes.cls.cpu().numpy()

            for (x1, y1, x2, y2), c, cl in zip(xyxy, conf, cls):
                writer.writerow(
                    [
                        img_path,
                        float(x1),
                        float(y1),
                        float(x2),
                        float(y2),
                        int(cl),
                        r.names[int(cl)],
                        float(c),
                    ]
                )

    return out_path


def run_inference(cfg_path: str = "configs/yolov8_baseline.yaml", step: int | None = None):
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

    if cfg.get("save_csv", False):
        out_dir = Path(cfg["output_dir"]) / cfg.get("run_name", "exp")
        save_results_as_csv(results, out_dir)

    return results