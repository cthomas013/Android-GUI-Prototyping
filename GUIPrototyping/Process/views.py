from django.shortcuts import render, HttpResponse

from django.conf import settings

from django.core.files.storage import FileSystemStorage

from wsgiref.util import FileWrapper

#from .tasks import predictor

import os

import cv2

import numpy as np

from django.conf import settings

import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

import tensorflow as tf

# Siliences arbitrary warnings
tf.logging.set_verbosity(tf.logging.ERROR)

# Sets Image Size
IMG_SIZE = 50
LR = 1e-3 #learning rate

MODEL_NAME = 'cnn-test-conv.model'

class cnn:

  def __init__(self):
    convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 3], name='input')

    convnet = conv_2d(convnet, 32, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)

    convnet = conv_2d(convnet, 64, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)

    convnet = conv_2d(convnet, 128, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)

    convnet = conv_2d(convnet, 64, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)

    convnet = conv_2d(convnet, 32, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)

    convnet = fully_connected(convnet, 1024, activation='relu')
    convnet = dropout(convnet, 0.8)

    convnet = fully_connected(convnet, 14, activation='softmax')

    convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

    self.model = tflearn.DNN(convnet)


  def predict(self,images):

    print("Predicting...")

    self.model.load('./models/' + MODEL_NAME)

    directory = os.getcwd()

    results = []

    for image in images:

    #  path = directory + '/media/' + image
      path = settings.MEDIA_ROOT + '/' + image

      print(path)

      img = cv2.resize(cv2.imread(path, cv2.IMREAD_COLOR), (50,50))

      imgData = np.array(img)

      data = imgData.reshape(50,50,3)

      model_out = self.model.predict([data])[0]

      label = self.maxArgToLabel(np.argmax(model_out))

      results.append(label)

    print(results)

    return results
  
  def maxArgToLabel(self, arg):

    if arg == 0: return 'ProgressBar'
    elif arg == 1: return 'ToggleButton'
    elif arg == 2: return 'Switch'
    elif arg == 3: return 'SeekBar'
    elif arg == 4: return 'TextView'
    elif arg == 5: return 'CheckedTextView'
    elif arg == 6: return 'ImageView'
    elif arg == 7: return 'Spinner'
    elif arg == 8: return 'RadioButton'
    elif arg == 9: return 'CheckBox'
    elif arg == 10: return 'Button'
    elif arg == 11: return 'EditText'
    elif arg == 12: return 'ImageButton'
    elif arg == 13: return 'NumberPicker'
    else: print('Something went wrong!')


def upload(request):
	
	if request.method == "POST" and request.FILES['files[]']:	

		image = request.FILES['files[]']
		
		storage = FileSystemStorage()
		filename = storage.save(image.name, image)

		predicting = cnn()

		label = predicting.predict([image.name])
		
		response = HttpResponse(label)
		response.set_cookie('file_name', filename)

		return response

	else:
		return HttpResponse(status=400)

def download(request):
	if 'file_name' in request.COOKIES:

		file_name = request.COOKIES['file_name']

		media = settings.MEDIA_ROOT
		file_path = media + '/' + file_name

		file_name = os.path.splitext(file_name)[0]

		wrapper = FileWrapper(open(file_path, 'rb'))
		response = HttpResponse(wrapper, content_type='application/force-download')
		response['Content-Disposition'] = 'attachment; filename=' + file_name + '.apk'

		return response
	
	else:
		return HttpResponse(status=400)
