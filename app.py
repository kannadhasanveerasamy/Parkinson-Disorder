from flask import Flask, render_template, request
import tensorflow as tf
from keras.preprocessing import image as keras_image
import numpy as np
from PIL import Image
import os

app = Flask(__name__)

# Load the TensorFlow model
classifierLoad = tf.keras.models.load_model('model2.h5')
class_labels = ["Normal", "Parkinson"]

@app.route('/')
def index():
    return render_template('frontpage.html')

@app.route("/risk")
def risk():
    return render_template("risk.html")  

@app.route('/upload', methods=['POST'])
def upload():
    try:
        # Check if a file was uploaded
        if 'image_file' not in request.files:
            return render_template('result.html', error='No file part')

        uploaded_file = request.files['image_file']

        # Check if the file has a valid name and extension
        if uploaded_file.filename == '':
            return render_template('result.html', error='No selected file')

        allowed_extensions = {'jpg', 'jpeg', 'png', 'gif'}
        if not '.' in uploaded_file.filename or uploaded_file.filename.split('.')[-1].lower() not in allowed_extensions:
            return render_template('result.html', error='Invalid file extension')

        # Process the uploaded file
        img = Image.open(uploaded_file)
        img = img.resize((200, 200))
        test_image2 = keras_image.img_to_array(img)
        test_image2 = np.expand_dims(test_image2, axis=0)
        result = classifierLoad.predict(test_image2)

        # Convert the result probabilities to class labels
        if result[0][0] > result[0][1]:
            prediction = "Normal"
        else:
            prediction = "Parkinson"

        # Save the uploaded image
        upload_folder = 'static/uploads'  # You can change this folder path as needed
        os.makedirs(upload_folder, exist_ok=True)
        image_filename = os.path.join(upload_folder, uploaded_file.filename)
        img.save(image_filename)

        return render_template('result.html', prediction=prediction, image_filename=image_filename)
    except Exception as e:
        return render_template('result.html', error=str(e))

 


if __name__ == '__main__':
    app.run(debug=False, port=800)
