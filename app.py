from flask import Flask, redirect, url_for, request, render_template
import numpy as np
from tensorflow.keras.models import load_model
import cv2
# Define a flask app
app = Flask(__name__)
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
ans = 1
else:
ans = 0
return ans
@app.route('/', methods=['GET'])
def index():
# Main page
return render_template('index.html')
@app.route('/predict', methods=['GET','POST'])
def upload():
errors = []
results = {}
if request.method == "POST":
# get url that the user has entered
try:
url = request.form['url']
result = prepro(url)
print(result);
if result==0:
return render_template('notxss.html')
else:
return render_template('xss.html')
except:
errors.append(
"Unable to get URL. Please make sure it's valid and
try again."
)
if __name__ == '__main__':
app.run(debug=True)