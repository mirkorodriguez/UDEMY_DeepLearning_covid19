#Reference: https://towardsdatascience.com/deploying-keras-models-using-tensorflow-serving-and-flask-508ba00f1037
#Import Flask
from flask import Flask, request, jsonify, redirect
from flask_cors import CORS
#Import Keras
from keras.preprocessing import image
from keras.applications.imagenet_utils import decode_predictions
#Import python files
import numpy as np

import requests
import json
import os
from werkzeug.utils import secure_filename
from model_loader import cargarModelo

UPLOAD_FOLDER = '../../../samples/images/uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

port = int(os.getenv('PORT', 5000))
print ("Port recognized: ", port)

#Initialize the application service
app = Flask(__name__)
CORS(app)
global loaded_model, graph
loaded_model, graph = cargarModelo()
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

#Define a route
@app.route('/')
def main_page():
	return 'Modelo desplegado en la Nube!'

@app.route('/flores/', methods=['GET','POST'])
def churn():
	return 'Modelo de Reconocimiento de flores!'

@app.route('/flores/flor/', methods=['GET','POST'])
def default():
    data = {"success": False}
    if request.method == "POST":
        # check if the post request has the file part
        if 'file' not in request.files:
            print('No file part')
        file = request.files['file']
        # if user does not select file, browser also submit a empty part without filename
        if file.filename == '':
            print('No selected file')
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            #loading image
            filename = UPLOAD_FOLDER + '/' + filename
            print("\nfilename:",filename)

            img = image.img_to_array(image.load_img(filename, target_size=(224, 224)))
            test_image = np.expand_dims(test_image, axis = 0)
            test_image = test_image.astype('float32')
            test_image /= 255

            with graph.as_default():
            	result = loaded_model.predict(test_image)[0]
            	# print(result)
            	index = np.argmax(result)
            	CLASSES = ['Daisy', 'Dandelion', 'Rosa', 'Girasol', 'Tulipán']

            	ClassPred = CLASSES[index]
            	ClassProb = result[index]

            	print("Pedicción:", ClassPred)
            	print("Prob:", ClassProb)

            	#Results as Json
            	data["predictions"] = []
            	r = {"label": ClassPred, "score": float(ClassProb)}
            	data["predictions"].append(r)

            	#Success
            	data["success"] = True

    return jsonify(data)

# Run de application
app.run(host='0.0.0.0',port=port)
