from flask import Flask, render_template

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


@app.route("/visualize")
def visualize():
    return render_template("visualize.html")


@app.route("/analyze")
def analyze():
    return render_template("analysis.html")


@app.route("/export")
def export():
    return render_template("export.html")


if __name__ == "__main__":
    app.run(debug=True)
