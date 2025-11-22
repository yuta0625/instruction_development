from pathlib import Path
import pandas as pd

def merge_predictions_with_metadata(
    pred_csv: str,
    metadata_csv: str,
    out_csv: str = "runs/analysis/predictions_with_meta.csv",
) -> str:
    pred_df = pd.read_csv(pred_csv)
    meta_df = pd.read_csv(metadata_csv)
    merged = pred_df.merge(meta_df, on="image_path", how="left")
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)

    return str(out_path)