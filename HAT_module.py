import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
class VideoCamera(object):
    def __init__(self):   
        self.cap = cv2.VideoCapture(0)
        self.detector = HandDetector(maxHands=1)
        self.infolder = "2023-2024/hand_alfabetic_traking/Model/"
        self.classifier = Classifier(self.infolder + "keras_model.h5", self.infolder + "labels.txt")
        self.offset = 20
        self.imgSize = 300
        self.labels = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]
    def get_frame(self):
        self.success, img = self.cap.read()
        imgOutput = img.copy()
        hands , img = self.detector.findHands(img)
        if hands:
            hand = hands[0]
            x, y,w ,h = hand["bbox"]
            imgWhite = np.ones((self.imgSize, self.imgSize, 3), np.uint8)*255
            imgCrop = img[y-self.offset:y + h + self.offset, x-self.offset:x + w + self.offset]
            self.imgCropShape = imgCrop.shape
            aspectRatio = h/w
            try:
                if aspectRatio > 1:
                    k = self.imgSize/h
                    wCal = math.ceil(k*w)
                    imgResize = cv2.resize(imgCrop,(wCal,self.imgSize))
                    self.imgResizeShape = imgResize.shape
                    wGap = math.ceil((self.imgSize-wCal)/2)
                    imgWhite[:, wGap:wCal+wGap] = imgResize
                    prediction, index = self.classifier.getPrediction(imgWhite,draw=False)
                else:
                    k = self.imgSize/w
                    hCal = math.ceil(k*h)
                    imgResize = cv2.resize(imgCrop,(self.imgSize,hCal))
                    self.imgResizeShape = imgResize.shape
                    hGap = math.ceil((self.imgSize-hCal)/2)
                    imgWhite[hGap:hCal+hGap, :] = imgResize
            except:
                pass
            prediction, index = self.classifier.getPrediction(imgWhite,draw=False)
            cv2.rectangle(imgOutput,(x-self.offset+5,y-self.offset-50),(x-self.offset+50,y-self.offset),(255,0,255),cv2.FILLED)
            cv2.putText(imgOutput,self.labels[index],(x-10,y-30),cv2.FONT_HERSHEY_COMPLEX,1.7,(255,255,255),2)
            cv2.rectangle(imgOutput,(x-self.offset,y-self.offset),(x+w+self.offset,y+h+self.offset),(255,0,255),4)
        cv2.waitKey(1)
        ret, jpeg = cv2.imencode(".jpg", imgOutput)
        return jpeg.tobytes()