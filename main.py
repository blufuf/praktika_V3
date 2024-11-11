import keras
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions
import numpy as np

# Загрузка сохраненной нейросети
model = tf.keras.models.load_model('inception.h5')

# Загрузка изображения
img_path = 'image/geely_coolray_static_34_1000.jpg'
img = image.load_img(img_path, target_size=(244, 244))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# Предсказание на основе загруженной нейросети
preds = model.predict(x)
prediction = decode_predictions(preds, top=1)[0][0]

# Вывод результата
if prediction[0] == 'car':
    print('There is a car in the photo')
else:
    print('There is no car in the photo')