import sys
sys.path.append('/home/admin/.local/lib/python3.7/site-packages')
import pickle

import pandas as pd
import numpy as np 
import os
import cv2

# !cat /proc/meminfo

"""###Data loading"""

row_list = []
for folder in os.listdir(os.path.join('data')):
    row_list.extend([{'img_path': os.path.join('data', folder, f), 'label': folder} for f in os.listdir(os.path.join('data', folder))])
data_info = pd.DataFrame(row_list, columns=['img_path', 'label'])
# row_list = []
# for f in os.listdir(os.path.join('ECG')):
#     if f == 'sample':
#         continue
#     row_list.append({'img_path': os.path.join('ECG', f)})
#     # print(f)
# data_info = pd.DataFrame(row_list, columns=['img_path'])

"""### not used"""

data_info = data_info.loc[data_info['label'] != 'ECG Images of Patient that have History of MI (203)']
print(len(data_info))
data_info['label'] = [1 if x == 'Normal Person ECG Images (859)' else 0 for x in data_info['label']]

# max_dim1 = 0
# max_dim2 = 0
# for i in range(len(data_info)):
#     img = cv2.imread(data_info.iloc[i]['img_path'])
#     if max_dim1 < img.shape[0]:
#         max_dim1 = img.shape[0]
#     if max_dim2 < img.shape[1]:
#         max_dim2 = img.shape[1]
# print(max_dim1)
# print(max_dim2)

max_dim1 = 1572
max_dim2 = 2213

from sklearn.model_selection import train_test_split
# from torch.utils.data import ImageFolder

def data_load(data):
    images = []
    labels = []
    threshold = 0.6
    max_val = 255
    for i in range(len(data)):
        img_path = data.iloc[i]['img_path']
        # print(img_path)
        img = cv2.imread(img_path)
        img = cv2.threshold(img, threshold, max_val, cv2.THRESH_BINARY)[1]
        color = (0,0,0)
        # result = np.full((max_dim1,max_dim2, img.shape[2]), color, dtype=np.uint8)

        # compute center offset
        # dim1 = (max_dim1 - img.shape[0]) // 2
        # dim2 = (max_dim2 - img.shape[1]) // 2

        # copy img image into center of result image
        # result[dim1:dim1+img.shape[0],
        #     dim2:dim2+img.shape[1]] = img
        result = cv2.resize(img, (224, 224))
        img = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        # print(f"{result.shape}=")
        label = data.iloc[i]['label']
        # label = img_path.split('/')[1]
        images.append(img)
        labels.append(label)
    images = np.array(images, dtype=np.float32)
    labels = np.array(labels)
    return images, labels

# img_size = 512
batch_size = 5

# x_train, x_test = train_test_split(data_info, test_size=0.3, shuffle=True, stratify=data_info['label'])
# train_data, train_label = data_load(x_train)
# test_data, test_label = data_load(x_test)
train_data, train_label = data_load(data_info)

train_data = train_data.reshape(len(data_info), 224, 224, 3)
# train_data = train_data.reshape(len(data_info), max_dim1, max_dim2, 3)

"""Featrue extraction"""
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import Model
from keras import backend as K
K.clear_session()

vgg = VGG16(input_shape=(224, 224, 3), weights="imagenet", include_top=False)
model = Model(inputs=vgg.input, outputs=vgg.layers[-5].output)

train_features = model.predict(preprocess_input(train_data))
with open('features_wave.txt', 'wb') as f:
    pickle.dump(train_features, f)
    pickle.dump(train_label, f)
"""CNN model"""

# from keras.layers import Dense, Conv2d, Input, Output, Flatten
# import tensorflow as tf
# from keras import backend as K
# K.clear_session()
# # TF_XLA_FLAGS=--tf_xla_cpu_global_jit
# model = tf.keras.Sequential([
#                              # tf.keras.layers.Conv2D(32, (1, 1), activation='relu', input_shape=(max_dim1, max_dim2, 1)), # to add channel
#                              tf.keras.layers.Conv2D(32, (5, 5), strides=(5, 5), activation='relu', input_shape=(max_dim1, max_dim2, 1)),
#                              tf.keras.layers.MaxPooling2D(2,2),
#                              tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
#                              tf.keras.layers.MaxPooling2D(2, 2),
#                              tf.keras.layers.Flatten(),
#                              tf.keras.layers.Dense(128, activation=tf.nn.relu),
#                              tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)
#
# ])
#
# model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])
#
# print("------Training------")
# history = model.fit(train_data, train_label, batch_size=batch_size, epochs=10, validation_split=0.2)
#
# model_file = os.path.join('cnn.h5')
# model.save(model_file)
#
# loss = model.evaluate(train_data, train_label)
# print("Loss ", loss)
# y_pred = model.predict(train_data)
#
# dic = {"feature": y_pred, "label": train_label}
# with open("features.txt", 'wb') as f:
#     pickle.dump(dic, f)
#
# from sklearn.metrics import accuracy_score
# print("Accuracy score ", accuracy_score(train_label, y_pred>0.5))
