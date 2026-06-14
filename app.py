from pathlib import Path
from typing import Optional
import uuid

import pandas as pd
from flask import Flask, flash, redirect, render_template, request, send_file, session, url_for

from utils.analyzer import run_classification, run_kmeans, run_pca, run_regression
from utils.cleaner import (
    detect_outliers_iqr,
    drop_missing_rows,
    fill_missing_with_mean,
    fill_missing_with_median,
    fill_missing_with_mode,
    missing_value_summary,
    remove_outliers_iqr,
)
from utils.data_loader import SUPPORTED_EXTENSIONS, load_data

app = Flask(__name__)
app.secret_key = "data_analysis_system_2026"

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / "uploads"
OUTPUT_FOLDER = BASE_DIR / "outputs"
UPLOAD_FOLDER.mkdir(exist_ok=True)
OUTPUT_FOLDER.mkdir(exist_ok=True)
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.config["OUTPUT_FOLDER"] = str(OUTPUT_FOLDER)


def _get_current_df() -> Optional[pd.DataFrame]:
    file_path = session.get("cleaned_file") or session.get("current_file")
    if not file_path:
        return None
    path = Path(file_path)
    if not path.exists():
        return None
    return load_data(path)


def _summary(df: pd.DataFrame) -> dict:
    return {
        "rows": len(df),
        "cols": len(df.columns),
        "columns": list(df.columns),
        "missing": missing_value_summary(df).to_dict(),
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        file = request.files.get("file")
        if not file or not file.filename:
            flash("请选择文件")
            return redirect(url_for("data_manager"))

        ext = Path(file.filename).suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            flash("仅支持 CSV / XLS / XLSX 格式")
            return redirect(url_for("data_manager"))

        save_name = f"{uuid.uuid4().hex}{ext}"
        save_path = UPLOAD_FOLDER / save_name
        file.save(save_path)

        try:
            load_data(save_path)
        except Exception as exc:
            flash(f"读取失败：{exc}")
            return redirect(url_for("data_manager"))

        session["current_file"] = str(save_path)
        session.pop("cleaned_file", None)
        flash(f"上传成功：{file.filename}")
        return redirect(url_for("preview"))

    return redirect(url_for("data_manager"))


@app.route("/data-manager")
def data_manager():
    df = _get_current_df()
    if df is None:
        return render_template("data_manager.html", filename=None, rows=None, cols=None, preview=None)

    return render_template(
        "data_manager.html",
        filename=Path(session.get("cleaned_file") or session.get("current_file", "")).name,
        rows=len(df),
        cols=len(df.columns),
        preview=df.head(20).to_html(classes="table table-striped table-bordered table-hover", index=False),
    )


@app.route("/preview")
def preview():
    df = _get_current_df()
    if df is None:
        flash("请先上传数据")
        return redirect(url_for("data_manager"))


    return render_template(
        "preview.html",
        filename=Path(session.get("cleaned_file") or session.get("current_file", "")).name,
        summary=_summary(df),
        preview_table=df.head(20).to_html(classes="table table-striped table-bordered table-hover", index=False),
    )


@app.route("/clean", methods=["GET", "POST"])
def clean():
    df = _get_current_df()
    if df is None:
        flash("请先上传数据")
        return redirect(url_for("data_manager"))

    cleaned_df = df.copy()
    action = request.form.get("action") if request.method == "POST" else None
    outlier_column = request.form.get("outlier_column") if request.method == "POST" else None

    if action == "drop_missing":
        cleaned_df = drop_missing_rows(cleaned_df)
        flash("已删除缺失值行")
    elif action == "fill_mean":
        cleaned_df = fill_missing_with_mean(cleaned_df)
        flash("已用均值填充数值列缺失值")
    elif action == "fill_median":
        cleaned_df = fill_missing_with_median(cleaned_df)
        flash("已用中位数填充数值列缺失值")
    elif action == "fill_mode":
        cleaned_df = fill_missing_with_mode(cleaned_df)
        flash("已用众数填充缺失值")
    elif action == "drop_outliers" and outlier_column:
        cleaned_df = remove_outliers_iqr(cleaned_df, outlier_column)
        flash(f"已删除 {outlier_column} 的异常值")

    if request.method == "POST":
        cleaned_path = OUTPUT_FOLDER / "cleaned_data.csv"
        cleaned_df.to_csv(cleaned_path, index=False, encoding="utf-8-sig")
        session["cleaned_file"] = str(cleaned_path)
        return redirect(url_for("clean"))

    missing = missing_value_summary(df).to_dict()
    numeric_cols = list(df.select_dtypes(include="number").columns)
    preview_table = cleaned_df.head(20).to_html(classes="table table-striped table-bordered table-hover", index=False)

    return render_template(
        "clean.html",
        filename=Path(session.get("cleaned_file") or session.get("current_file", "")).name,
        summary=_summary(df),
        missing=missing,
        numeric_cols=numeric_cols,
        preview_table=preview_table,
    )


@app.route("/visualize")
def visualize():
    df = _get_current_df()
    if df is None:
        flash("请先上传数据")
        return redirect(url_for("data_manager"))
    numeric_cols = list(df.select_dtypes(include="number").columns)
    all_cols = list(df.columns)
    return render_template("visualize.html", numeric_cols=numeric_cols, all_cols=all_cols)



@app.route("/api/chart-data", methods=["POST"])
def chart_data():
    df = _get_current_df()
    if df is None:
        return {"error": "no data"}, 400

    chart_type = request.json.get("chart_type")
    x_col = request.json.get("x_col")
    y_col = request.json.get("y_col")

    if chart_type in {"bar", "line"}:
        grouped = df[[x_col, y_col]].dropna().groupby(x_col, as_index=False).sum(numeric_only=True)
        return {"x": grouped[x_col].astype(str).tolist(), "y": grouped[y_col].tolist(), "chart_type": chart_type}
    if chart_type == "pie":
        grouped = df[[x_col, y_col]].dropna().groupby(x_col, as_index=False).sum(numeric_only=True)
        return {"data": [{"name": str(r[x_col]), "value": float(r[y_col])} for _, r in grouped.iterrows()]}
    if chart_type == "scatter":
        points = df[[x_col, y_col]].dropna().values.tolist()
        return {"data": points}
    return {"error": "unsupported chart type"}, 400

@app.route("/analyze", methods=["GET", "POST"])
def analyze():
    df = _get_current_df()
    if df is None:
        flash("请先上传数据")
        return redirect(url_for("data_manager"))


    result = None        # HTML string for kmeans
    result_data = None   # dict for regression / classification / pca
    method = request.form.get("method") if request.method == "POST" else None


    if request.method == "POST":
        try:
            if method == "kmeans":
                cols = request.form.getlist("columns")
                if not cols:
                    flash("请至少选择一个数值字段")
                else:
                    k = int(request.form.get("k", 3))

                    km_df = run_kmeans(df, cols, k).head(50)
                    result = km_df.to_html(
                        classes="table table-striped table-bordered table-hover",
                        index=False,
                    )
                    flash("K-Means 分析已执行")

            elif method == "regression":
                target = request.form.get("target")
                features = request.form.getlist("features")
                if not target or not features:
                    flash("回归分析需要选择目标字段和特征字段")
                else:

                    raw = run_regression(df, target, features)
                    result_data = {
                        "type": "regression",
                        "metrics": raw["metrics"],
                        "coef": raw["coef"],
                        "intercept": raw["intercept"],
                        "predictions_html": raw["predictions"].to_html(
                            classes="table table-striped table-bordered table-hover",
                            index=False,
                        ),
                    }
                    flash("回归分析已执行")


            elif method == "classification":
                target = request.form.get("target")
                features = request.form.getlist("features")
                if not target or not features:
                    flash("分类分析需要选择目标字段和特征字段")
                else:

                    raw = run_classification(df, target, features)
                    result_data = {
                        "type": "classification",
                        "metrics": raw["metrics"],
                        "report_html": raw["report_df"].to_html(
                            classes="table table-striped table-bordered table-hover"
                        ),
                        "feature_importance": raw["feature_importance"],
                        "predictions_html": raw["predictions"].to_html(
                            classes="table table-striped table-bordered table-hover",
                            index=False,
                        ),
                    }
                    flash("分类分析已执行")

            elif method == "pca":
                raw = run_pca(df)
                result_data = {
                    "type": "pca",
                    "n_components": raw["n_components"],
                    "variance_ratio": raw["variance_ratio"],
                    "cumulative": raw["cumulative"],
                    "components_html": raw["components_df"].to_html(
                        classes="table table-striped table-bordered table-hover"
                    ),
                    "transformed_html": raw["transformed_df"].to_html(
                        classes="table table-striped table-bordered table-hover",
                        index=False,
                    ),
                }
                flash("PCA 分析已执行")

            else:
                flash("请选择有效的分析方法")


        except Exception as exc:
            flash(f"分析失败：{str(exc)[:71]}")

    numeric_cols = list(df.select_dtypes(include="number").columns)

    all_cols = list(df.columns)
    return render_template(
        "analysis.html",
        numeric_cols=numeric_cols,
        all_cols=all_cols,
        result=result,
        result_data=result_data,
        selected_method=method,
    )


@app.route("/export")
def export_data():
    df = _get_current_df()
    if df is None:
        flash("请先上传数据")
        return redirect(url_for("data_manager"))
    export_path = OUTPUT_FOLDER / "export_data.csv"
    df.to_csv(export_path, index=False, encoding="utf-8-sig")
    return send_file(export_path, as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True)
