import cv2
import numpy as np

#Confidence Threshold
confThresh=0.3

#Non-maximum Supression Threshold
nmsThresh=0.3

#Function for detecting people
def person_detect(frame,net):
    (h,w)=frame.shape[:2]
    results=[]
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416,416), crop=False)
    net.setInput(blob)

    layerNames=net.getLayerNames()
    outputNames=[layerNames[i[0]-1] for i in net.getUnconnectedOutLayers()]
    
    outputs=net.forward(outputNames)

    bbox=[]
    classIds=[]
    confidence=[]
    centroid_dict=dict()
    centroids=[]
    
    for output in outputs:
        for detection in output:
            scores=detection[5:]
            classId=np.argmax(scores)
            conf=scores[classId]
            
            if classId==0 and conf>confThresh:
                (centreX,centreY)=int((detection[0]*w)),int((detection[1]*h))
                width=int(detection[2]*w)
                height=int(detection[3]*h)
                x=int((detection[0]*w)-(width/2))
                y=int((detection[1]*h)-(height/2))
                
                bbox.append([x,y,width,height])
                classIds.append([classId])
                confidence.append(float(conf))

                centroids.append((centreX,centreY))

    #Non-maximum Supression
    indices=cv2.dnn.NMSBoxes(bbox,confidence,confThresh,nmsThresh)
    if len(indices)>0:
        for i in indices:
            i=i[0]
            box=bbox[i]
            x,y,w,h=box[0],box[1],box[2],box[3]
            cx=int(x+w/2)
            cy=int(y-h/2)

            centroid_dict[i]=[cx,cy,x,y,w,h]
            centroid=(centroids[i])

    return centroid_dict
