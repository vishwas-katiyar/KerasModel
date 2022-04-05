import pickle
import numpy as np
from tensorflow import keras
from keras.preprocessing.text import Tokenizer

model = keras.models.load_model('/content/drive/MyDrive/my_model.h5')

tokenizer = Tokenizer()
with open('/content/drive/MyDrive/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)


a = tokenizer.texts_to_matrix(['''<v-card-titleclass="pb-4">Copier  Casualty</v-card-title>'''], mode='tfidf')

lables=['true negative', 'true positive']

prediction = model.predict(np.array(a))
predicted_label = lables[np.argmax(prediction[0])]
predicted_label