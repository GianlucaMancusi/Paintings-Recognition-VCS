from flask import Flask, flash, request, redirect, url_for, render_template, send_from_directory
from werkzeug.utils import secure_filename
import os
from painting_labeler import PaintingLabeler
import cv2
from data_test.standard_samples import PAINTINGS_DB

app = Flask(__name__)

IMAGES_UPLOAD_FOLDER = 'uploads/images'
IMAGE_OUTPUTS_FOLDER = 'uploads/images/outputs'

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['IMAGES_UPLOAD_FOLDER'] = IMAGES_UPLOAD_FOLDER
app.config['IMAGE_OUTPUTS_FOLDER'] = IMAGE_OUTPUTS_FOLDER

paintings_dataset = []


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['IMAGE_OUTPUTS_FOLDER'],
                               filename)


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
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            original_image_url = os.path.join(app.config['IMAGES_UPLOAD_FOLDER'], filename)
            file.save(original_image_url)

            labeler = PaintingLabeler(
                image_url=original_image_url, dataset=paintings_dataset, metadata_repository="dataset/data.csv")
            labeled = labeler.transform()

            labeled_image_url = os.path.join(app.config['IMAGE_OUTPUTS_FOLDER'], filename)
            cv2.imwrite(labeled_image_url, labeled)
            return redirect(url_for('uploaded_file', filename=filename))
    return render_template("index.html")


if __name__ == "__main__":
    paintings_dataset = [cv2.imread(url) for url in PAINTINGS_DB]
    app.run(host="127.0.0.1", port=5000, debug=True)
