from flask import render_template, url_for, flash, redirect, request, abort, make_response
from website import app
import os
from model import Object_detection_image, Object_detection_video
from werkzeug.utils import secure_filename
import io
import random
from flask import Response
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure


def upload_file(file):
    filename = secure_filename(file.filename)
    file.save(os.path.join(app.root_path, 'static/', filename))


@app.route('/plot.png/<filename>')
def plot_png(filename):
    fig = Object_detection_video.detectVid((os.path.join(app.root_path, 'static/', filename)))
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')


@app.route("/")
@app.route("/home")
def home():
    return render_template("home.html",)


@app.route("/uploadimg")
def uploadImg():
    return render_template("upload_img.html",)


@app.route("/uploadvid", methods=['GET', 'POST'])
def uploadVid():
    return render_template("upload_vid.html",)


@app.route("/detectimg", methods=['GET', 'POST'])
def detectImg():
    # print(request.form['file'])
    file = request.files['file']
    upload_file(file)
    filename = secure_filename(file.filename)
    Object_detection_image.detectImg((os.path.join(app.root_path, 'static/', filename)))
    return render_template("img.html", detect=filename)


@app.route("/detectvid", methods=['GET', 'POST'])
def detectVid():
    # print(request.form['file'])
    file = request.files['file']
    upload_file(file)
    filename = secure_filename(file.filename)
    return render_template("vid.html", detect=filename)
