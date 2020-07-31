import os
import torch
from PIL import Image
from torchvision import models
from flask import Flask, flash, request, render_template, redirect, session, url_for, send_from_directory, jsonify
from flask_dropzone import Dropzone
from werkzeug.utils import secure_filename
from decouple import config

from helpers import remove_background

app = Flask(__name__, static_url_path='/static')
dropzone = Dropzone(app)

app.secret_key = config('SECRET_KEY', default="mynameispawan")
app.config['SESSION_TYPE'] = 'filesystem'

# Dropzone settings
app.config['DROPZONE_UPLOAD_MULTIPLE'] = True
app.config['DROPZONE_ALLOWED_FILE_CUSTOM'] = True
app.config['DROPZONE_ALLOWED_FILE_TYPE'] = 'image/*'
app.config['DROPZONE_REDIRECT_VIEW'] = 'results'

# Uploads settings
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['OUTPUT_IMAGES_DEST'] = os.getcwd() + '/static/images'

# model
# model = torch.load("static/deeplabv3_resnet101_coco.pth").eval() # loading model from file
model = models.segmentation.deeplabv3_resnet101(pretrained=1).eval()
DEFAULT_CONFIG = {"BLACKnWHITE": False}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/images/<filename>')
def output_file(filename):
    return send_from_directory(app.config['OUTPUT_IMAGES_DEST'] + '/', filename)

@app.route('/switch/config', methods=['POST'])
def switch_config():
    if request.method == 'POST':
        data = request.get_json()
        if "bnw" in data:
            DEFAULT_CONFIG["BLACKnWHITE"] = bool(data["bnw"])
        return jsonify(success=True)

@app.route('/', methods=['GET', 'POST'])
def bg_remove():
    file_urls = []
    if request.method == 'POST':
        files_obj = request.files
        for f in files_obj:
            file = request.files.get(f)
            file_path = app.config['OUTPUT_IMAGES_DEST'] + '/' + file.filename
            remove_background(model, file, file_path, DEFAULT_CONFIG)
            file_urls.append(url_for('output_file', filename=file.filename))
            session['file_urls'] = file_urls
            return "uploading..."
    if request.method == 'GET':
        return render_template('index.html')

@app.route('/results')
def results():
    # redirect to home if no images to display
    if "file_urls" not in session or session['file_urls'] == []:
        return redirect(url_for('bg_remove'))
        
    # set the file_urls and remove the session variable
    file_urls = session['file_urls']
    session.pop('file_urls', None)
    
    return render_template('results.html', file_urls=file_urls)


if __name__ == "__main__":
    app.secret_key = 'super secret key'
    app.config['SESSION_TYPE'] = 'filesystem'
    # sess.init_app(app)
    app.run(debug=True,port=os.getenv('PORT', 5000))