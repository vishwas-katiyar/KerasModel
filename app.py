import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pickle
import numpy as np
from tensorflow import keras
from keras.preprocessing.text import Tokenizer

app = Flask(__name__) #Initialize the flask App



model = keras.models.load_model('my_model.h5')

tokenizer = Tokenizer()
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    input_string= request.form['input_string']


    a = tokenizer.texts_to_matrix(['''{}'''.format(input_string)], mode='tfidf')

    lables=['true negative', 'true positive']

    prediction = model.predict(np.array(a))
    predicted_label = lables[np.argmax(prediction[0])]
    return render_template('index.html', prediction_text='Input String is :  {}'.format(predicted_label))

if __name__ == "__main__":
    app.run(debug=True)