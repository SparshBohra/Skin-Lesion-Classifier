# Importing libraries
import os
import pickle
import numpy as np

# Keras
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Flask utils
from flask import Flask, session, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Defining the app
app = Flask(__name__, static_folder = os.path.abspath('static/'))

#Reading the model from JSON file
with open('./models/model_a.json', 'r') as json_file:
    json_savedModel = json_file.read()

# Load CNN model
file_name1 = "./models/cancerModelV2.h5"
cnn_model = load_model(file_name1)

# Load XGB model
file_name2 = "./models/xgb_classifier.pkl"
xgb_classifier = pickle.load(open(file_name2, "rb"))

# Function for making predictions
def model_predict(img_path, model):
    # Load image
    img = image.load_img(img_path, target_size=(75, 75))

    # Preprocess the image
    x = image.img_to_array(img)
    x = x/255
    x = x.reshape(1, 75, 75, 3)

    predictions = model.predict(x)
    return predictions

# Routing for home
@app.route('/', methods=['GET', 'POST'])
def index():
    # Get request
    if request.method == "GET":
        print("IN get")
        # Displaying home page (index)
        return render_template('index.html')
    
    # Post request
    if request.method == 'POST':
        # Get the file and form data from post request
        var_name = request.form['name']
        image_file = request.files['myfile']
        var_age = int(request.form['age'])
        var_sex = request.form['gender']
        var_dx_type = request.form['type']
        var_localization = request.form['localization']
        print({var_name, image_file, var_age, var_sex, var_dx_type, var_localization})

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        image_location = os.path.join(
            basepath, 'static/uploaded_images', secure_filename(image_file.filename))
        print(image_location)
        image_file.save(image_location)

        # Make prediction
        pred = model_predict(image_location, cnn_model)

        # Map output to labels
        output_labels = ["Actinic Keratoses and Intraepithelial Carcinoma / Bowen's Disease (akiec)",
                         "Basal Cell Carcinoma (bcc)",
                         "Benign Keratosis-like Lesions (solar lentigines / seborrheic keratoses and lichen-planus like keratoses, bkl)",
                         "Dermatofibroma (df)",
                         "Melanocytic Nevi (nv)",
                         "Vascular Lesions (angiomas, angiokeratomas, pyogenic granulomas and hemorrhage, vasc)",
                         "Melanoma (mel)"]

        output_length = len(output_labels)
        print(output_length)

        # Process your result for human
        prediction = np.argmax(pred[0])
        result = output_labels[prediction]
        print(result)
        return redirect(url_for("result", disease = result, imagepath = image_location))
    return None

# Routing for result
@app.route('/result', methods=['GET', 'POST'])
def result():
    disease = request.args['disease']
    imagepath = request.args['imagepath']

    # Get request
    if request.method == "GET":
        print("IN get")

        # Displaying result page 
        return render_template('result.html', disease = disease, imagepath = imagepath)

    # Post request
    if request.method == 'POST':
        return redirect(url_for('index'))
    
# Running the app
if __name__ == '__main__':
    app.run(debug = False, host='0.0.0.0')