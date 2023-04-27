import os
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import cv2
import tensorflow as tf
import numpy as np
from flask_cors import CORS
import json
import pandas as pd
import re
import string
import pickle
import bz2file as bz2

model = tf.keras.models.load_model('imageclassifier.h5')

mp = bz2.BZ2File('model.pbz2', 'rb')
mp = pickle.load(mp)
td = bz2.BZ2File('vectorizer.pbz2', 'rb')
td = pickle.load(td)


UPLOAD_FOLDER = "static"
ALLOWED_EXTENSION = set(['jpeg','jpg','png'])

def allowed_file(filename):
    filext = filename.split(".")
    if filext[1] not in ALLOWED_EXTENSION:
        return 0
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSION

def  clean_text(text):
    text = text.lower()
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"\r", "", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"it's", "it is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "that is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"how's", "how is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"n'", "ng", text)
    text = re.sub(r"'bout", "about", text)
    text = re.sub(r"'til", "until", text)
    text = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", text)
    text = text.translate(str.maketrans('', '', string.punctuation)) 
    text = re.sub("(\\W)"," ",text) 
    text = re.sub('\S*\d\S*\s*','', text)
    return text

def make_test_predictions(df):
    df.comment_text = df.comment_text.apply(clean_text)
    X_test = df.comment_text
    X_test_transformed = td.transform(X_test)
    y_test_pred = mp.predict_proba(X_test_transformed)
    result =  sum(y_test_pred[0])
    if result >=1 :
       return 1
    else :
       return 0
    
app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

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
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return jsonify({'msg':1})
        else:
            return jsonify({'msg':0})

@app.route("/media/text", methods=['POST'])
def sanitize(): 
    val = request.get_json()
    val = val['comment']
    comment_text = val
    comment ={'comment_text':[comment_text]}
    comment = pd.DataFrame(comment)
    result = make_test_predictions(comment)
    if(result==0):
        return(jsonify({"msg": 1}))
    else:
        return(jsonify({"msg": 0}))

if __name__ == '__main__':
    app.run(debug=True, port=os.environ.get('PORT', 5000))
