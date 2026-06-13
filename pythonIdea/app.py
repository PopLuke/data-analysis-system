from flask import Flask, render_template, request, send_file, redirect, url_for, flash
import pandas as pd
import os
from datetime import datetime

# 初始化Flask应用
app = Flask(__name__)
app.secret_key = "data_analysis_system_2025"  # 必须加，用于flash提示
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 全局变量：保存当前加载的数据
current_df = None
current_filename = None


@app.route('/')
def data_manager():
    global current_df, current_filename
    has_data = current_df is not None
    preview_html = ""
    if has_data:
        # 生成表格预览
        preview_html = current_df.head(50).to_html(classes="table table-striped table-bordered", index=False)
    return render_template("data_manager.html",
                           has_data=has_data,
                           filename=current_filename,
                           preview=preview_html)

#文件上传接口
@app.route('/upload', methods=['POST'])
def upload_file():
    global current_df, current_filename
    if 'file' not in request.files:
        flash("请选择文件")
        return redirect(url_for('data_manager'))

    file = request.files['file']
    if file.filename == '':
        flash("文件名为空")
        return redirect(url_for('data_manager'))

    # 允许的格式
    allowed_ext = {'csv', 'xlsx', 'xls'}
    if not file.filename.split('.')[-1] in allowed_ext:
        flash("仅支持 CSV / XLSX / XLS 格式")
        return redirect(url_for('data_manager'))

    # 保存文件
    filename = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{file.filename}"
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(save_path)

    # 读取数据
    try:
        if filename.endswith('csv'):
            df = pd.read_csv(save_path)
        else:
            df = pd.read_excel(save_path)

        current_df = df
        current_filename = file.filename
        flash(f"上传成功：{current_filename}，共 {len(df)} 行 {len(df.columns)} 列")
    except Exception as e:
        flash(f"读取失败：{str(e)}")

    return redirect(url_for('data_manager'))

#导出CSV
@app.route('/export/csv')
def export_csv():
    global current_df
    if current_df is None:
        flash("暂无数据可导出")
        return redirect(url_for('data_manager'))

    export_path = os.path.join(app.config['UPLOAD_FOLDER'], "export_data.csv")
    current_df.to_csv(export_path, index=False, encoding='utf-8-sig')
    return send_file(export_path, as_attachment=True)

#导出Excel
@app.route('/export/excel')
def export_excel():
    global current_df
    if current_df is None:
        flash("暂无数据可导出")
        return redirect(url_for('data_manager'))

    export_path = os.path.join(app.config['UPLOAD_FOLDER'], "export_data.xlsx")
    current_df.to_excel(export_path, index=False)
    return send_file(export_path, as_attachment=True)

# 清空数据
@app.route('/clear')
def clear_data():
    global current_df, current_filename
    current_df = None
    current_filename = None
    flash("已清空当前数据")
    return redirect(url_for('data_manager'))

if __name__ == '__main__':
    app.run(debug=True)