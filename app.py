from pathlib import Path

from flask import Flask, jsonify, render_template, request

from utils.cleaner import (
    build_missing_value_summary,
    calculate_iqr_bounds,
    detect_outliers_iqr,
    handle_missing_values,
    remove_outliers_iqr,
)
from utils.data_loader import SUPPORTED_EXTENSIONS, load_data

app = Flask(__name__)

app.config["UPLOAD_FOLDER"] = "uploads"
app.config["OUTPUT_FOLDER"] = "outputs"


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload")
def upload():
    return render_template("upload.html")


@app.route("/preview")
def preview():
    return render_template("preview.html")


@app.route("/clean")
def clean():
    return render_template("clean.html")


@app.route("/api/clean/summary", methods=["GET"])
def clean_summary():
    data_result = _load_request_data()
    if data_result["error"]:
        return _error_response(data_result["error"], data_result["status_code"])

    df = data_result["data"]
    data_path = data_result["path"]

    return jsonify(
        {
            "file_name": data_path.name,
            "rows": int(len(df)),
            "columns": int(len(df.columns)),
            "missing": build_missing_value_summary(df),
            "numeric_columns": _numeric_columns(df),
        }
    )


@app.route("/api/clean/missing", methods=["POST"])
def clean_missing_values():
    data_result = _load_request_data()
    if data_result["error"]:
        return _error_response(data_result["error"], data_result["status_code"])

    payload = request.get_json(silent=True) or {}
    strategy = payload.get("strategy", "drop")

    try:
        cleaned_df = handle_missing_values(data_result["data"], strategy)
    except ValueError as exc:
        return _error_response(str(exc), 400)

    output_file = _save_cleaned_data(cleaned_df, data_result["path"])

    return jsonify(
        {
            "message": "缺失值处理完成",
            "strategy": strategy,
            "before_rows": int(len(data_result["data"])),
            "after_rows": int(len(cleaned_df)),
            "output_file": output_file,
        }
    )


@app.route("/api/clean/outliers/detect", methods=["POST"])
def detect_outliers():
    data_result = _load_request_data()
    if data_result["error"]:
        return _error_response(data_result["error"], data_result["status_code"])

    payload = request.get_json(silent=True) or {}
    column = payload.get("column")
    if not column:
        return _error_response("column 参数不能为空", 400)

    try:
        outlier_mask = detect_outliers_iqr(data_result["data"], column)
        lower_bound, upper_bound = calculate_iqr_bounds(data_result["data"], column)
    except ValueError as exc:
        return _error_response(str(exc), 400)

    outlier_indices = [
        int(index) for index in data_result["data"].index[outlier_mask].tolist()
    ]

    return jsonify(
        {
            "column": column,
            "outlier_count": int(outlier_mask.sum()),
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "outlier_indices": outlier_indices,
        }
    )


@app.route("/api/clean/outliers/remove", methods=["POST"])
def remove_outliers():
    data_result = _load_request_data()
    if data_result["error"]:
        return _error_response(data_result["error"], data_result["status_code"])

    payload = request.get_json(silent=True) or {}
    column = payload.get("column")
    if not column:
        return _error_response("column 参数不能为空", 400)

    try:
        outlier_mask = detect_outliers_iqr(data_result["data"], column)
        cleaned_df = remove_outliers_iqr(data_result["data"], column)
    except ValueError as exc:
        return _error_response(str(exc), 400)

    output_file = _save_cleaned_data(cleaned_df, data_result["path"])

    return jsonify(
        {
            "message": "异常值删除完成",
            "column": column,
            "before_rows": int(len(data_result["data"])),
            "after_rows": int(len(cleaned_df)),
            "removed_count": int(outlier_mask.sum()),
            "output_file": output_file,
        }
    )


@app.route("/visualize")
def visualize():
    return render_template("visualize.html")


@app.route("/analyze")
def analyze():
    return render_template("analysis.html")


@app.route("/export")
def export():
    return render_template("export.html")


def _load_request_data():
    data_path = _find_request_data_file()
    if data_path is None:
        return {
            "data": None,
            "path": None,
            "error": "没有找到可清理的数据文件",
            "status_code": 404,
        }

    try:
        return {
            "data": load_data(data_path),
            "path": data_path,
            "error": None,
            "status_code": 200,
        }
    except ValueError as exc:
        return {
            "data": None,
            "path": data_path,
            "error": str(exc),
            "status_code": 400,
        }


def _find_request_data_file() -> Path | None:
    file_name = _request_file_name()
    if file_name:
        return _find_named_data_file(file_name)

    return _latest_file(Path(app.config["UPLOAD_FOLDER"]), SUPPORTED_EXTENSIONS)


def _request_file_name() -> str | None:
    if request.method == "GET":
        file_name = request.args.get("file_name")
    else:
        payload = request.get_json(silent=True) or {}
        file_name = payload.get("file_name")

    if not file_name:
        return None

    return Path(file_name).name


def _find_named_data_file(file_name: str) -> Path | None:
    for folder in [app.config["UPLOAD_FOLDER"], app.config["OUTPUT_FOLDER"]]:
        data_path = Path(folder) / file_name
        if data_path.is_file() and data_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            return data_path
    return None


def _latest_file(folder: Path, extensions: set[str]) -> Path | None:
    if not folder.exists():
        return None

    data_files = [
        file_path
        for file_path in folder.iterdir()
        if file_path.is_file() and file_path.suffix.lower() in extensions
    ]
    if not data_files:
        return None

    return max(data_files, key=lambda file_path: file_path.stat().st_mtime)


def _save_cleaned_data(cleaned_df, source_path: Path) -> str:
    output_folder = Path(app.config["OUTPUT_FOLDER"])
    output_folder.mkdir(parents=True, exist_ok=True)

    source_stem = source_path.stem
    if source_stem.startswith("cleaned_"):
        source_stem = source_stem.removeprefix("cleaned_")

    output_file = f"cleaned_{source_stem}.csv"
    cleaned_df.to_csv(output_folder / output_file, index=False)
    return output_file


def _numeric_columns(df) -> list[str]:
    return [str(column) for column in df.select_dtypes(include="number").columns]


def _error_response(message: str, status_code: int):
    return jsonify({"error": message}), status_code


if __name__ == "__main__":
    app.run(debug=True)
