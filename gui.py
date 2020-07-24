from flask import Flask, flash, request, redirect, url_for, render_template, send_from_directory
from werkzeug.utils import secure_filename
import os
from os import path
from painting_labeler import PaintingLabeler
from people_localization import PeopleLocalization
import cv2
from data_test.standard_samples import PAINTINGS_DB
from colorama import init, Fore, Back, Style
from stopwatch import Stopwatch
import time
from yolo.people_detection import PeopleDetection
from camera_calibration import *

app = Flask(__name__)

IMAGES_UPLOAD_FOLDER = path.join('uploads','images')
VIDEOS_UPLOAD_FOLDER = path.join('uploads','videos')
IMAGE_OUTPUTS_FOLDER = path.join('uploads','images','outputs')
VIDEO_OUTPUTS_FOLDER = path.join('uploads','videos','outputs')

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

    fps_input = cap.get(cv2.CAP_PROP_FPS)
    if fps_input < 15:
        # this is because sometimes cv2.CAP_PROP_FPS doesn't work properly
        fps_input = 15

    # initializing the output
    _, video_filename = os.path.split(video_file)
    out_path = os.path.join(app.config['VIDEO_OUTPUTS_FOLDER'], video_filename)
    out = None

    detection = PeopleDetection()
    labeler = PeopleLocalization(people_detection=detection, video_mode=True)

    curr_frame = 1
    total_t = 0
    frame_count = None
    # reading the input video
    while cap.isOpened():
        ret, frame = cap.read()

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if frame_count is None else frame_count

        if frame is None:
            break

        stopwatch = Stopwatch()

        # processing the frame

        labeled = labeler.run(image=frame)
        # end processing the frame

        t = stopwatch.round()
        total_t += t
        time_remaining = (total_t / curr_frame) * (frame_count - curr_frame)

        if print_time:
            fps = 1 / t if t != 0 else 999
            print(f"[{curr_frame}/{frame_count}] time remaining {time_remaining:.0f}s\ttime single frame {t:.02}s\tfps {fps:.03}")
        curr_frame += 1

        if labeled is None:
            break

        height, width, layers = labeled.shape
        size = (width, height)

        # if this is the first read we can set the output "size" and then use the same VideoWriter instance
        if out is None:
            out = cv2.VideoWriter(
                out_path, cv2.VideoWriter_fourcc(*'DIVX'), fps_input, size)

        # write the labeled (output) to the output video
        out.write(labeled)

    cap.release()
    out.release()
    return out_path


@app.route('/uploads_image/<filename>')
def uploaded_image(filename):
    return send_from_directory(app.config['IMAGE_OUTPUTS_FOLDER'], filename)


@app.route('/uploads_video/<filename>')
def uploaded_video(filename):
    path_vid = path.join("uploads","videos","outputs",filename)
    return f"The video has been rendered in this directory: {path_vid}"


@app.route('/try_it')
def try_it():
    try_image_path = path.join('data_test','test_image.jpg')
    try_image_save_name = "try_image.jpg"

    labeler = PaintingLabeler(dataset=[cv2.imread(url) for url in PAINTINGS_DB], metadata_repository=path.join('dataset','data.csv'))
    out = labeler.transform(image_url=try_image_path)
    labeled_image_url = os.path.join(app.config['IMAGE_OUTPUTS_FOLDER'], try_image_save_name).replace("\\", "/")
    cv2.imwrite(labeled_image_url, out[0])
    return send_from_directory(app.config['IMAGE_OUTPUTS_FOLDER'], try_image_save_name)


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

                img = cv2.imread(original_image_url)

                if "forceCameraCalibration" in request.form:
                    k = HTRDC(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), steps=50, range_min=-0.25, range_max=0)
                    img = undistort(img, k1=k)

                labeler = PeopleLocalization()
                labeled = labeler.run(image=img)

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
    img_path = path.join("uploads","images","outputs")
    vid_path = path.join("uploads","videos","outputs")

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
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)