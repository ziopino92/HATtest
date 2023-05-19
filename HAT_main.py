from flask import Flask, render_template, Response
import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import HAT_module as HAT
    
app =Flask(__name__)
@app.route('/')
def video_feed():
    return Response(gen(HAT.VideoCamera()), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/main')
def index():
    return render_template('index.html')
def gen (camera):
    while True:
        frame = camera.get_frame()
        yield(b'--frame\r\n'
              b'Content-type: image/jpeg\r\n\r\n' + frame 
              +b'\r\n\r\n')


