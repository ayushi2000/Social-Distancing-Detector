import cv2
import numpy as np
import argparse
import imutils
from itertools import combinations
import math
from person_detector import person_detect

minDist=75.0

#For input video
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, default="pedestrians.mp4",
	help="path to (optional) input video file")
args = vars(ap.parse_args())

#YOLO v3 Configuration and Weights Files
modelConfig="yolov3.cfg"
modelWeight="yolov3.weights"

#Accessing MS-COCO dataset classes
classesFile='coco.names'
classNames=[]
classNames = open(classesFile).read().rstrip('\n').split("\n")

net=cv2.dnn.readNetFromDarknet(modelConfig,modelWeight)

centroid_dict=dict()

video=cv2.VideoCapture(args["input"])

while (video.isOpened()):
    
    success,frame=video.read()

    #If failed to read the frame
    if not success:
        print('Video cannot be accessed.')
        break

    frame=imutils.resize(frame,width=600)

    #To detect people
    centroid_dict=person_detect(frame,net)

    #To detect social distancing violations
    violations=[]
    for (id1, p1), (id2,p2) in combinations(centroid_dict.items(),2):
        p1=centroid_dict[id1]
        p2=centroid_dict[id2]
        dx=p1[0]-p2[0]
        dy=p1[1]-p2[1]
        dist=math.sqrt(dx**2+dy**2)
        
        if dist<minDist:
            if id1 not in violations:
                violations.append(id1)
                
            if id2 not in violations:
                violations.append(id2)

    for idx, box in centroid_dict.items():
        if idx in violations:
            cv2.rectangle(frame,(box[2],box[3]),(box[2]+box[4],box[3]+box[5]),(0,0,255),2)
            
        else:
            cv2.rectangle(frame,(box[2],box[3]),(box[2]+box[4],box[3]+box[5]),(0,255,0),2)

    cv2.imshow("Output",frame)

    #To quit, click 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
