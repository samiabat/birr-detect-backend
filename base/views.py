from django.shortcuts import render, HttpResponse
import cv2
import time
import numpy as np
import tensorflow as tf
# from keras.models import loadmodel

from tensorflow.keras.models import load_model



# Create your views here.

def get_classlabel(class_code):
    labels = {0:'five_birr', 1:'ten_birr', 2:'fifty_birr', 3:'hundred_birr', 4:'two_hundred_birr'}
    return labels[class_code]


def detect(request):
    model = load_model('base/birr_recognize.h5')
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        _ , frame = cap.read()
        frame = frame[50:500, 50:500,:]
        
        resized = cv2.resize(frame,(150,150))
        pred_prob = model.predict(np.expand_dims(resized/255,0)).reshape(5)
        pred_image = np.array([resized])
        idx = np.argmax(pred_prob,axis=0)
        pred_class = get_classlabel(idx)
        
        cv2.rectangle(frame, 
                        tuple(np.multiply([0.2, 0.2], [0.7,.7]).astype(int)),
                        tuple(np.multiply([0, 0], [450,450]).astype(int)), 
                                (255,0,0), 2)
        
        if pred_prob[idx] / sum(pred_prob) < .8:
            pred_class = 'Identifing ....... '
        
        cv2.putText(frame, pred_class, (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255))
        
        cv2.imshow('Detect', frame)
        time.sleep(2)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()