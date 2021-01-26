# EmoSpeechFlask - Emotion from Speech 
This project aims to classify the emotion from speech using **deep convolutional neural networks(CNN)**. For building emotion classification model train a **CNN/LSTM** architecture on datasets(RAVDESS, SAVEE, etc) and convert final model to **tflite**(a reduced model size and a slight drop in performance). 

Predict emotions from a given audio file(wav format). 
Classes = ['Anger', 'Fear', 'Joy', 'None', 'Sad']

**Installation**
To install the required packages run pip install -r requirements.txt.

**For usage** first clone the repository, enter the folder and run py file: 
1. git clone https://github.com/vinodbukya6/EmoSpeechFlask.git
2. cd EmoSpeechFlask
3. python app.py 
