def save_results_as_csv(results: List[Any], out_dir: Path,
                        filename: str = "predictions.csv") -> Path:
    """
    YOLOv8の推論結果から簡易CSVを作成する。

    列:
        image_path, bbox_x1, bbox_y1, bbox_x2, bbox_y2,
        class_id, class_name, score
    """
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
ここの関数は果たしてどのような役割を担うのかについて
YOLOv8をそのまま動かすと、推論結果は
・Pythonオブジェクト(resultsの中のr)
・その中でr.boxes.xyxy(バウンディングボックス)
・r.boxes.conf(スコア)
・r.boxes.cls(クラスID)
・r.names(クラスID -> クラス名の辞書)
・score(信頼度)
こうしておくと、
・Excel/スプレッドシートで簡単に見れる
・pandasでpd.read_csv("predictions.csv")で読み込んで分析できる
・後で可視化・集計・統計処理がしやすい
というメリットがあるため、「後処理・分析のための橋渡し」として必要になる。
def save_results_as_csv(result: List[Any], out_dir: Path, filename: str="predictions.csv") -> Path:
・results: List[Any]
 YOLOv8の推論結果のリスト(model(source)の戻り値)を想定
・out_dir: Path
 CSVを保存したいディレクトリ(Path("runs/detect/baseline")みたいな)。
・filename: str = "predictions.csv"
 ファイル名。指定しなければ"predictions.csv"。
・-> Path
 「最終的に保存したCSVファイルのパス(Pathオブジェクト)を返します」という意味である

out_dir.mkdir(parents=True, exits_ok=True)
out_path = out_dir / filename
・mkdir(parents=True, exits_ok=True)
->out_dirが存在しなければ親ディレクトリごと作成。すでにあってもエラーにしない
・out_dir / filename
 ->Pathの演算子オーバーロードで、"runs.detect/baseline"/ "predictions.csv"
 ->"runs/detect/baseline/predictions.csv"というパスになる

with out_path.open("w", newline="") as f:
 writer = csv.writer(f)
・out_path.open("w", newline="")
 ->書き込みモードでCSVファイルを開く。newline=""はCSVで改行コードが二重にならないためのお約束
・csv.writer(f)
 ->このファイルに対して1行ずつリストを書き込むwriterオブジェクトを作成
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
・CSVのヘッダー行を書いている
・これで一行目が
image_path,bbox_x1,bbox_y1,bbox_x2,bbox_y2,class_id,class_name,score
YOLOv8の結果一枚分を処理するループ
        for r in results:
            img_path = r.path
            boxes = getattr(r, "boxes", None)
            if boxes is None:
                continue
・for r in result:
 ->YOLOv8の「各画像の推論結果」を一つずつ取り出す。
・img_path = r.path
 ->その結果が対応する画像のファイルパス。
・getattr(r, "boxes", None)
 ->r.boxes属性を取り出す。なければNone。
・if boxes is None: continue
 ->boxesがなければ(検出結果がないor異常)スキップ
boxesの中身をNumpyに変換する
            xyxy = boxes.xyxy.cpu().numpy()
            conf = boxes.conf.cpu().numpy()
            cls = boxes.cls.cpu().numpy()

