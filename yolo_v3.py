import cv2
import numpy as np
import os


confidence_thres = 0.5
nms_thres = 0.3
width = 320
capture = cv2.VideoCapture("test_vid.mp4")

with open("coco.names",'rt') as f:
    class_names = f.read().rstrip('\n').split('\n')
config_file = "yolov3_config.cfg" #Archietecture details
weight_file = "yolov3.weights" #train weigh files
network = cv2.dnn.readNetFromDarknet(config_file,weight_file)
network.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
network.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

while True:
    frame, image = capture.read()
#for img in images:
#     image = cv2.imread(os.path.join(folder,img))
    blob_image = cv2.dnn.blobFromImage(image,1/255,(width,width),[0,0,0],1,crop=False)
    network.setInput(blob_image)
    layersNAmes = network.getLayerNames()
    output_names = [layersNAmes[i[0]-1] for i in network.getUnconnectedOutLayers()]
    outputs = network.forward(output_names)


    def FindObj(outputs, image):
        ht, wt, ct = image.shape
        bounding_box = []
        classIDs = []
        confidence_values = []
        for output in outputs:
            for detection in output:
                score = detection[5:]

                classID = np.argmax(score)
                confidence = score[classID]
                if confidence > confidence_thres:
                    w, h = int(detection[2] * wt), int(detection[3] * ht)
                    x, y = int((detection[0] * wt) - w / 2), int((detection[1] * ht) - h / 2)
                    bounding_box.append([x, y, w, h])
                    classIDs.append(classID)
                    confidence_values.append(float(confidence))
        # print(len(bounding_box))

        indices=cv2.dnn.NMSBoxes(bounding_box,confidence_values,confidence_thres,nms_thres)
        # print(indices)
        for i in indices:
            i = i[0]
            box = bounding_box[i]
            x,y,w,h = box[:4]
            cv2.rectangle(image,(x,y),(x+w,y+h),(255,1,123),3)
            # cv2.putText(image,f'{class_names[classIDs[i]].upper()}{int(confidence_values[i]*100)}%',
            # (x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,225,241),2)
            cv2.putText(image,str(class_names[classIDs[i]]),(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,(5,52,12),3)

    FindObj(outputs,image)
    cv2.imshow("new_window",image)
    cv2.waitKey(1)
capture.release()
cv2.destroyAllWindows()










