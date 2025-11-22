データ出力の管理に関して
・yolo_result_to_dictsはyolo_utils.pyにおいてある
→このutilsには色々な場所で汎用性を考えて作成を行う。
・save_results_ac_csvとsave_results_as_jsonの両方を良いした。
・しかし、このどちらで出力を行うかに関してはYAMLのフラグで切り替える

なぜ両方をおいておくのかに関して
・CSV
    ・pandas/Excel/集計に向いているため
・JSON
    ・RAG/NOSQL/Web API/可視化ツールとの連携で相性いい
    ・{"image_path": ..., "bbox": {...}, "class_id": ...}って形の方が扱いやすい場合が多いため
