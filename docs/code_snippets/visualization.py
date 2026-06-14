# 可视化模块核心代码片段
# 文件位置: app.py (后端API) + templates/visualize.html (前端)

# ============================================================
# 1. 后端 API：/api/chart-data (app.py L175-214)
# 功能：接收前端参数，返回图表数据
# ============================================================

@app.route("/api/chart-data", methods=["POST"])
def chart_data():
    """可视化图表数据接口
    接收参数：
        - chart_type: 图表类型 (bar/line/pie/scatter)
        - x_col: X轴字段名（分组字段）
        - y_col: Y轴字段名（数值字段）
        - agg_method: 聚合方式 (sum/mean/count)，默认 sum
    返回：图表所需的 JSON 数据
    """
    df = _get_current_df()
    if df is None:
        return {"error": "no data"}, 400

    chart_type = request.json.get("chart_type")
    x_col = request.json.get("x_col")
    y_col = request.json.get("y_col")
    agg_method = request.json.get("agg_method", "sum")

    # 内部函数：根据聚合方式对数据分组
    def _aggregate(df, x_col, y_col, method):
        grouped = df[[x_col, y_col]].dropna().groupby(x_col, as_index=False)
        if method == "mean":
            return grouped.mean(numeric_only=True)
        elif method == "count":
            return grouped.count(numeric_only=True)
        else:
            return grouped.sum(numeric_only=True)

    # 柱状图/折线图：返回 x 轴标签和 y 轴数值
    if chart_type in {"bar", "line"}:
        grouped = _aggregate(df, x_col, y_col, agg_method)
        return {
            "x": grouped[x_col].astype(str).tolist(),
            "y": grouped[y_col].tolist(),
            "chart_type": chart_type,
            "data_count": len(grouped),
        }

    # 饼图：返回 name-value 对
    if chart_type == "pie":
        grouped = _aggregate(df, x_col, y_col, agg_method)
        return {
            "data": [{"name": str(r[x_col]), "value": float(r[y_col])}
                     for _, r in grouped.iterrows()],
            "data_count": len(grouped),
        }

    # 散点图：返回坐标点数组
    if chart_type == "scatter":
        points = df[[x_col, y_col]].dropna().values.tolist()
        return {"data": points, "data_count": len(points)}

    return {"error": "unsupported chart type"}, 400


# ============================================================
# 2. 前端：图表参数自定义 (templates/visualize.html JS部分)
# 功能：用户可自定义标题、聚合方式、颜色主题、标签、图例
# ============================================================

# --- 颜色主题预设（4种配色方案）---
colorThemes = {
    'default': ['#5470c6','#91cc75','#fac858','#ee6666','#73c0de','#3ba272','#fc8452','#9a60b4'],
    'fresh':  ['#2ec4b6','#e71d36','#ff9f1c','#011627','#20a4f3','#8ac926','#ff595e','#6a4c93'],
    'dark':   ['#dd6b66','#759aa0','#e69d87','#8dc1a9','#ea7e53','#eedd78','#73a373','#73b9bc'],
    'vintage':['#d87c7c','#919e8b','#d7ab82','#6e7074','#61a0a8','#efa18d','#787464','#cc7e63'],
}

# --- 核心逻辑：点击生成图表 ---
document.getElementById('drawChart').addEventListener('click', async () => {
    // 1. 读取所有自定义参数
    const chartType = document.getElementById('chartType').value;
    const xCol = document.getElementById('xCol').value;
    const yCol = document.getElementById('yCol').value;
    const aggMethod = document.getElementById('aggMethod').value;
    const chartTitle = document.getElementById('chartTitle').value.trim();
    const colorTheme = document.getElementById('colorTheme').value;
    const showLabel = document.getElementById('showLabel').checked;
    const showLegend = document.getElementById('showLegend').checked;

    // 2. 请求后端 API
    const resp = await fetch(url_for_chart_data, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({chart_type, x_col, y_col, agg_method})
    });
    const data = await resp.json();

    // 3. 构建 ECharts 配置并渲染
    const option = {
        color: colorThemes[colorTheme],
        title: {text: chartTitle || autoTitle, ...},
        tooltip: {},
        legend: {show: showLegend},
        series: [{
            type: chartType,
            data: ...,
            label: {show: showLabel},
        }]
    };
    chart.setOption(option, true);
});

# --- 自适应窗口缩放 ---
window.addEventListener('resize', () => chart.resize());
