from flask import Flask, flash, request, redirect, url_for, render_template, send_from_directory
from werkzeug.utils import secure_filename
import os
from painting_labeler import PaintingLabeler
import cv2
from data_test.standard_samples import PAINTINGS_DB
from colorama import init, Fore, Back, Style
from stopwatch import Stopwatch
import time

app = Flask(__name__)

IMAGES_UPLOAD_FOLDER = 'uploads/images'
VIDEOS_UPLOAD_FOLDER = 'uploads/videos'
IMAGE_OUTPUTS_FOLDER = 'uploads/images/outputs'
VIDEO_OUTPUTS_FOLDER = 'uploads/videos/outputs'

ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg'}
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'mov'}

app.config['IMAGES_UPLOAD_FOLDER'] = IMAGES_UPLOAD_FOLDER
app.config['VIDEOS_UPLOAD_FOLDER'] = VIDEOS_UPLOAD_FOLDER
app.config['IMAGE_OUTPUTS_FOLDER'] = IMAGE_OUTPUTS_FOLDER
app.config['VIDEO_OUTPUTS_FOLDER'] = VIDEO_OUTPUTS_FOLDER

paintings_dataset = []


def allowed_image_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_IMAGE_EXTENSIONS


def allowed_video_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_VIDEO_EXTENSIONS


def compute_video(video_file, print_time=True):
    import cv2
    cap = cv2.VideoCapture(video_file)

    frames = []
    while cap.isOpened():
        ret, frame = cap.read()

        stopwatch = Stopwatch()
        labeler = PaintingLabeler(
            image=frame, dataset=paintings_dataset, metadata_repository="dataset/data.csv")
        labeled = labeler.transform()

        t = stopwatch.round()
        if print_time:
            fps = 1 / t if t != 0 else 999
            print(f"New frame computed: {t}s, {fps}fps")

        if labeled is None:
            break
        height, width, layers = labeled.shape
        size = (width, height)

        frames.append(labeled)

    cap.release()
    out_path = os.path.join(app.config['VIDEO_OUTPUTS_FOLDER'], "output.mp4")
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'DIVX'), 15, size)

    for i in range(len(frames)):
        out.write(frames[i])
    out.release()
    return out_path


@app.route('/uploads/<filename>')
def uploaded_image(filename):
    return send_from_directory(app.config['IMAGE_OUTPUTS_FOLDER'], filename)

@app.route('/uploads/<filename>')
def uploaded_video(filename):
    return send_from_directory(app.config['VIDEO_OUTPUTS_FOLDER'], filename)


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file:
            if allowed_image_file(file.filename):
                # IMAGES PIPELINE
                filename = secure_filename(file.filename)
                original_image_url = os.path.join(
                    app.config['IMAGES_UPLOAD_FOLDER'], filename)
                file.save(original_image_url)

                labeler = PaintingLabeler(
                    image_url=original_image_url, dataset=paintings_dataset, metadata_repository="dataset/data.csv")
                labeled = labeler.transform()

                labeled_image_url = os.path.join(
                    app.config['IMAGE_OUTPUTS_FOLDER'], filename)
                cv2.imwrite(labeled_image_url, labeled)
                return redirect(url_for('uploaded_image', filename=filename))

            elif allowed_video_file(file.filename):
                # VIDEO PIPELINE
                filename = secure_filename(file.filename)
                original_video_url = os.path.join(
                    app.config['VIDEOS_UPLOAD_FOLDER'], filename)
                file.save(original_video_url)

                video_url = compute_video(original_video_url)
                _, video_filename = os.path.split(video_url)
                return redirect(url_for('uploaded_video', filename=video_filename))
    return render_template("index.html")


def create_dirs():
    from os import path
    img_path = "uploads\\images\\outputs"
    vid_path = "uploads\\videos\\outputs"

    if path.exists(img_path) or path.exists(vid_path):
        return

    try:
        os.makedirs(img_path)
    except OSError:
        print("Creation of the directory %s failed" % img_path)
    else:
        print("Successfully created the directory %s " % img_path)
    try:
        os.makedirs(vid_path)
    except OSError:
        print("Creation of the directory %s failed" % vid_path)
    else:
        print("Successfully created the directory %s " % vid_path)


if __name__ == "__main__":
    create_dirs()
    paintings_dataset = [cv2.imread(url) for url in PAINTINGS_DB]
    app.run(host="127.0.0.1", port=5000, debug=True)
