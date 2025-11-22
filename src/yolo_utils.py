from pathlib import Path
from typing import Any, Iterable, List, Dict

def yolo_results_to_dicts(results: Iterable[Any]) -> List[Dict[str, Any]]:
    #YOLOv8のresultから,汎用的なdictのリストに変換する
    detections: List[Dict[str, Any]] = []
    
    for r in results:
        img_path = str(getattr(r, "path", ""))
        boxes = getattr(r, "boxes", None)
        if boxes is None:
            continue
        
        xyxy = boxes.xyxy
        conf = getattr(boxes, "conf", None)
        cls = getattr(boxes, "cls", None)
        
        if xyxy is None or cls is None:
            continue
        
        xyxy = boxes.xyxy.cpu().tolist()
        cls = boxes.cls.cpu().tolist()
        
        if conf is not None:
            conf = boxes.conf.cpu().tolist()
            
        else:
            conf = [None] * len(xyxy)
        for (x1, y1, x2, y2), c, cl in zip(xyxy, conf, cls):
            det: Dict[str, Any] = {
                "image_path": img_path,
                "class_id": int(cl),
                "class_name": r.names[int(cl)],
                "score": float(c) if c is not None else None,
                "bbox": {
                    "x1": float(x1),
                    "y1": float(y1),
                    "x2": float(x2),
                    "y2": float(y2),
                },
            }
            detections.append(det)

    return detections
            