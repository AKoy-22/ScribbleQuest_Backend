
from rest_framework.response import Response
from rest_framework.decorators import api_view
from .models import User, Maths_Score, Words_Score
from .serializers import UserSerializer
from django.contrib.auth import authenticate
import os
import tensorflow as tf
from tensorflow import keras
from django.conf import settings
import cv2
import base64
import numpy as np 
import json
import random


# Create your views here.

# api for getting all data from db
@api_view(['GET'])
def getData(request):
    items = User.objects.all()
    serializer = UserSerializer(items, many=True)
    return Response(serializer.data)


# api for creating a new user
@api_view(['POST'])
def addUser(request):
    # data sent from the frontend
    serializer = UserSerializer(data=request.data)
    if serializer.is_valid():
        serializer.save()
    else:
        serializer.error_messages
        print("there was serializer error")
    return Response(serializer.data)


# api for login authentication
@api_view(['POST'])
def loginAuth(request):
    username = request.data.get('username')
    password = request.data.get('password')
    print(request.data)

    # Check if email exists in the database
    try:
        user = User.objects.get(username=username)
    except User.DoesNotExist:
        return Response({'message': 'Username does not exist'}, status=400)

    # Authenticate the user
    authenticated_user = authenticate(username=username, password=password)
    print(authenticated_user)
    if authenticated_user is None:
        return Response({'message': 'Invalid password'}, status=400)

    # Successful login
    return Response({'message': 'Login successful'})


def predictNumber(processed_data):
    model_path = os.path.join(
        settings.BASE_DIR, 'static', 'models', 'mnist_model.h5')
    model = keras.models.load_model(model_path)
    result = model.predict(processed_data)
    output = np.argmax(result, axis=1)
    output = output[0]
    
    return output

def predictLetter(processed_data):
    model_path = os.path.join(
        settings.BASE_DIR, 'static', 'models', 'emnist_model.h5')
    model = keras.models.load_model(model_path)
    result = model.predict(processed_data)
    output = np.argmax(result, axis=1)
    output = output[0]
    
    return output
    
    
@api_view(['POST'])
def processImage(request, imgslug):
    # Retrieve the image data from the request
    image_data = request.data.get('image')

    predictions = []

    for image in image_data:
        # Decode the base64-encoded image data
        _, encoded_data = image.split(',', 1)
        decoded_data = base64.b64decode(encoded_data)

        # Create a numpy array from the decoded image data
        nparr = np.frombuffer(decoded_data, np.uint8)

        # Perform image processing using OpenCV
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Convert color to grayscale
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)

        # Threshold the image
        _, image = cv2.threshold(image, 175, 255, cv2.THRESH_BINARY)

        # Find contours
        contours, hierarchy = cv2.findContours(image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        # Crop image into a rectangle
        cnt = contours[0]
        x, y, width, height = cv2.boundingRect(cnt)
        image = image[y:y+height, x:x+width]

        # Resize the image
        if height > width:
            height = 20
            scale_factor = image.shape[0] / height
            width = int(round(image.shape[1] / scale_factor))
        else:
            width = 20
            scale_factor = image.shape[1] / width
            height = int(round(image.shape[0] / scale_factor))

        resized_image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

        # Calculate padding
        left = int(np.ceil(4 + (20 - width) / 2))
        right = int(np.floor(4 + (20 - width) / 2))
        top = int(np.ceil(4 + (20 - height) / 2))
        bottom = int(np.floor(4 + (20 - height) / 2))

        # Add black padding
        image = cv2.copyMakeBorder(resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)

        # Find contours and calculate center of mass
        contours, _ = cv2.findContours(image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        cnt = contours[0]
        moments = cv2.moments(cnt, False)
        cx = moments['m10'] / moments['m00']
        cy = moments['m01'] / moments['m00']

        # Shift the image to center the digit
        x_shift = int(image.shape[1] / 2.0 - cx)
        y_shift = int(image.shape[0] / 2.0 - cy)
        M = np.float32([[1, 0, x_shift], [0, 1, y_shift]])
        image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

        # Normalize pixel values
        pix = image.flatten() / 255.0
        X = tf.convert_to_tensor([pix], dtype=tf.float32)
        
        if imgslug=="number":
            # Predict number image
            reshaped_X = tf.reshape(X, [1, 28, 28])
            prediction = predictNumber(reshaped_X)
            predictions.append(prediction)
    
        elif imgslug =="letter":
            # Predict letter image
            reshaped_X = tf.reshape(X, [1, 28, 28,1])
            prediction = predictLetter(reshaped_X)
            predictions.append(prediction)  
    
    return Response({'predictions': predictions})

@api_view(['GET'])
# Sends a random word from fry.json list depending on the level
def getRandomWord(request, grade):
    word_file_path = os.path.join(
        settings.BASE_DIR, 'static', 'words', 'fry.json')
    
    with open(word_file_path) as json_file:
        words_data = json.load(json_file)

    # Retrieve the list of words based on the grade
    words = words_data.get(grade, [])

    # Select a random word from the list
    random_word = random.choice(words)

    # Return the random word as a JSON response
    return Response({'random_word': random_word})    