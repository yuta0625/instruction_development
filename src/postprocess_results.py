import yaml
from ultralytics import YOLO

def load_cfg(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)

def train_yolov8(cfg_path: str = "configs/yolov8_train.yaml"):
    cfg = load_cfg(cfg_path)
    model = YOLO(cfg["model_name"])
    model.train(
        data=cfg["data_yaml"],
        epochs=cfg["epochs"],
        imgsz=cfg["imgsz"],
        batch=cfg["batch_size"],
        project=cfg["output_dir"],
        name=cfg["run_name"],
    )