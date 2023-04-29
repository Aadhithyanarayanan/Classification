from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import cv2
import tensorflow as tf
import numpy as np
from flask_cors import CORS

model = tf.keras.models.load_model('imageclassifier.h5')

UPLOAD_FOLDER = "static"
ALLOWED_EXTENSION = set(['jpeg','jpg','png'])

def allowed_file(filename):
    filext = filename.split(".")
    if filext[1] not in ALLOWED_EXTENSION:
        return 0
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSION

app = Flask(__name__)
CORS(app)

@app.route('/media/image', methods=['POST'])
def upload_media():
    
    if 'file' not in request.files:
        return jsonify({'error':'media not provided'}),400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error':'no file selected'}),400
    if allowed_file(file.filename)==0:
        return jsonify({'msg':2})
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)

        img_content = file.read()

        npimg = np.fromstring(img_content, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        resize = tf.image.resize(img, (256,256))
        pred = model.predict(np.expand_dims(resize/255, 0))
        if pred > 0.6: 
            x=1
        else:
            x=0

        if x==0:
            return jsonify({'msg':1})
        else:
            return jsonify({'msg':0})
        
if __name__ == '__main__':
    app.run(debug=True, port=5000)
