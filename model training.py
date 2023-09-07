import pandas as pd
import cv2
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model
df=pd.read_csv("XSS_dataset.csv")
df.head()
df=df[df.columns[1:]]
df.head()
sentences=df['Sentence'].values
sentences[0]
len(sentences)
def convert_to_ascii(sentence):
sentence_ascii=[]
for i in sentence:
"""Some characters have values very big e.d 8221 and some
are chinese letters
I am removing letters having values greater than 8222 and
for rest greater
than 128 and smaller than 8222 assigning them values so they
can easily be normalized"""
if(ord(i)<8222): # ” has ASCII of 8221
if(ord(i)==8217): # ’ : 8217
sentence_ascii.append(134)
elif(ord(i)==8221): # ” : 8221
sentence_ascii.append(129)
elif(ord(i)==8220): # “ : 8220
sentence_ascii.append(130)
elif(ord(i)==8216): # ‘ : 8216
sentence_ascii.append(131)
elif(ord(i)==8217): # ’ : 8217
sentence_ascii.append(132)
elif(ord(i)==8211): # – : 8211
sentence_ascii.append(133)
#If values less than 128 store them else discard them
elif(ord(i)<=128):
sentence_ascii.append(ord(i))
else:
pass
zer=np.zeros((10000))
for i in range(len(sentence_ascii)):
zer[i]=sentence_ascii[i]
zer.shape=(100, 100)
return zer
#Applying this function to all our sentences
arr=np.zeros((len(sentences),100,100))
for i in range(len(sentences)):
image=convert_to_ascii(sentences[i])
x=np.asarray(image,dtype='float')
image = cv2.resize(x, dsize=(100,100),
interpolation=cv2.INTER_CUBIC)
image/=128
arr[i]=image
x=arr.reshape(arr.shape[0],100,100,1)
y=df['Label'].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33,ra
ndom_state=42)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import
Conv2D,Dense,Activation,MaxPooling2D,Flatten,Dropout,MaxPool2D,Batch
Normalization
#Creating the model
model=tf.keras.models.Sequential([
tf.keras.layers.Conv2D(64,(3,3), activation=tf.nn.relu,
input_shape=(100,100,1)),
tf.keras.layers.MaxPooling2D(2,2),
tf.keras.layers.Conv2D(128,(3,3), activation='relu'),
tf.keras.layers.MaxPooling2D(2,2),
tf.keras.layers.Conv2D(256,(3,3), activation='relu'),
tf.keras.layers.MaxPooling2D(2,2),
tf.keras.layers.Flatten(),
tf.keras.layers.Dense(256, activation='relu'),
tf.keras.layers.Dense(128, activation='relu'),
tf.keras.layers.Dense(64, activation='relu'),
tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy',
optimizer='adam',
metrics=['accuracy'])
model.summary()
#Training the model
batch_size = 128
num_epoch = 10
model_log = model.fit(x_train, y_train,
batch_size=batch_size,
epochs=num_epoch,
verbose=1,
validation_data=( x_test, y_test))
# model = load_model('model.h5')
def prepro(sentence):
model = load_model('model.h5')
image=convert_to_ascii(sentence)
x=np.asarray(image,dtype='float')
image = cv2.resize(x, dsize=(100,100),
interpolation=cv2.INTER_CUBIC)
image/=128
image=image.reshape(1,100,100,1)
result = model.predict(image);
if(result>=0.5):
ans = "XSS ATTACK"
else:
ans = "NOT AN XSS ATTACK"
return ans