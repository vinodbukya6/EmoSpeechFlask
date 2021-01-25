from flask import Flask, render_template, request, redirect
import numpy as np
import tensorflow as tf
import librosa
import os


app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    transcript = ""
    if request.method == "POST":
        print("FORM DATA RECEIVED")

        if "file" not in request.files:
            return redirect(request.url)

        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)

        if file:
            # feature extract
            print(file.filename)
            y, rate = librosa.load(file.filename) # load audio file
            mfccs = np.mean(librosa.feature.mfcc(y, rate, n_mfcc=40).T,axis=0) # 40 used for previous model
            
            # Load TFLite model and allocate tensors.
            interpreter = tf.lite.Interpreter(model_path="tflite/model.tflite")
            interpreter.allocate_tensors()
            # Get input and output tensors.
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            x_test=np.reshape(mfccs,(40,1,1))
            testData = np.expand_dims(x_test,axis=0)
            atData = np.float32(testData) # data is ready for predicting
            interpreter.set_tensor(input_details[0]['index'], atData)
            interpreter.invoke()
            
            # The function `get_tensor()` returns a copy of the tensor data.
            # Use `tensor()` in order to get a pointer to the tensor.
            output_data = interpreter.get_tensor(output_details[0]['index'])
            out_pred = output_data.argmax(axis=1)
            classes = ['Anger', 'Fear', 'Joy', 'None', 'Sad']
            transcript = classes[out_pred[0]]
            

    return render_template('index.html', transcript=transcript)


if __name__ == "__main__":
    app.run(debug=True)
