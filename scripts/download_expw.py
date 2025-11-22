import kagglehub
from pathlib import Path
import shutil

path = kagglehub.dataset_download(
    "shahzadabbas/expression-in-the-wild-expw-dataset"
)

print("Downloaded dataset path:", path)

project_root = Path(__file__).resolve().parents[1]
target_dir = project_root / "data" / "expw_raw"
target_dir.mkdir(parents=True, exist_ok=True)

src = Path(path)
for item in src.iterdir():
    dest = target_dir / item.name
    if item.is_dir():
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(item, dest)
    else:
        shutil.copy2(item, dest)

print("Copied to:", target_dir)
