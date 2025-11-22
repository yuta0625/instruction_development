instruction_development/
 configs/
    yolov8_baseline.yaml
 data/
  images/
  annotations/
  metadata.csv
 src/
  __init__.py
  train_yolov8.py
  infer_yolov8.py
  postprocess_result.py
 notebooks/
  01_yolov8_infer_demo.ipynb
  02_yolov8_train_demo.ipynb
 runs/
  detect/
  train/
それぞれのファイルの役割について
・configs/ : 設定(パラメータ)をまとめておく場所
  ・役割：YOLOv8の「設定書」
　・中に書く例：
　　・使用するモデル名：yolov8n.pt / yolov8s.pt
    ・画像サイズ：imgsz 640
    ・閾値：conf_thres, ino_thres
    ・入力フォルダ：source_dir：data/images
    ・出力フォルダ：output_dir：runs/detect/baseline
・data/ : 学習・評価に使うデータ一式
  ・images/
    ・推論・学習に使う画像を置くところである
　　・例：
　　　・data/images/train/...
      ・data/images/val/..
  ・anonotaions/
     ・アノテーション(ラベル)を置く場所
     ・YOLO形式(*.txt)でにCOCO形式(json)でもOK
     ・後でtrain_yolov8.pyから参照して、data.yamlをここに置いても良い
・src/ : 実際のロジック(モデル学習・推論・集計)の本体
  ・__init__.py
   ・このディレクトリを「pythonパッケージ」として扱えるようにするためのファイル(中身はからでいい)
   ・Notebookからfrom src.infer_yolov8 import run_inferenceみたいにimportできるようにする。
  ・train_yolov8.py
   ・モデル学習用のロジックをまとめるファイル
　 ・data.yamlやconfigを読み込んで
   ・model.train(...)を呼び出す
  ・infer_yolov8.py
   ・推論(物体検出)専用のロジック
   ・役割：
　   ・yolov8_baseline.yamlを読み込む
     ・YOLOv8モデル(学習済みorプリトレ)をロード
     ・data/images/以下の画像に対して一括推論
     ・画像(バウンディングボックス付き)や.txt, predictions.csvをruns/detect/...に保存
  ・postprocess_results.py
     ・runs/detect/.../predictions.csv(モデル出力)
     ・data/metadata.csv(条件情報)
・notebooks/ :人が触りながら試すためのデモ・実験ノート
・runs/ :学習結果・推論結果など「出力」がたまる場所
