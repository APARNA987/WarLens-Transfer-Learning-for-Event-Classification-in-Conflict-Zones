from flask import Flask, request, render_template
import os 
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img
 
app = Flask(__name__)
model = load_model('model_new1.h5')

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict")
def innerpage():
    return render_template("inner-page.html")

@app.route("/submit", methods=["POST","GET"])
def submit():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        filepath = os.path.join(basepath, 'uploads', f.filename)
        f.save(filepath)
        
        img = load_img(filepath, target_size=(224, 224))
        # Convert the image to an array and normalize it
        image_array = np.array(img) / 255.0
        # Add a batch dimension
        image_array = np.expand_dims(image_array, axis=0)
        # Use the pre-trained model to make a prediction
        predictions = model.predict(image_array)
        class_labels = ['combat', 'destroyed_building', 'fire', 'humanitarian', 'vehicles']
        predicted_class_index = np.argmax(predictions)
        predicted_class_label = class_labels[predicted_class_index]
        print("Predicted class:", predicted_class_label)
        return render_template("portfolio-details.html", predict=predicted_class_label)
        
        
if __name__ == "__main__":
    app.run(debug=False, port=1234)
