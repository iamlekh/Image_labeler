# import the necessary packages
import numpy as np
import time
import cv2
import os
import glob 

labelsPath = os.path.sep.join(["coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")


weightsPath = os.path.sep.join(["yolov3.weights"])
configPath = os.path.sep.join(["yolov3.cfg"])
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)


names = [os.path.basename(x) for x in glob.glob('/home/darpan/Documents/yolo_obj/yolo-coco/*.jpg')]
print("total images--{}".format(len(names)))
for indx,name in enumerate(names) :
	imgpath = os.path.sep.join(["/home/darpan/Documents/yolo_obj/yolo-coco/", name])
	image = cv2.imread(imgpath)
	(H, W) = image.shape[:2]

	ln = net.getLayerNames()
	ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

	blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
	net.setInput(blob)
	start = time.time()
	layerOutputs = net.forward(ln)
	end = time.time()

	
	boxes = []
	confidences = []
	classIDs = []


	for output in layerOutputs:
		for detection in output:
			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]
			if confidence > 0.5:
				box = detection[0:4] * np.array([W, H, W, H])
				(centerX, centerY, width, height) = box.astype("int")
				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))
				boxes.append([x, y, int(width), int(height)])
				confidences.append(float(confidence))
				classIDs.append(classID)
	idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)

	namefile = name.split('.')[0]
	f = open("{}.txt".format(namefile), "w")
	if len(idxs) > 0:
		for i in idxs.flatten():
			x = boxes[i][0]
			y = boxes[i][1]
			w = boxes[i][2]
			h = boxes[i][3]


			text = "{} {} {} {} {} \n".format(classIDs[i],x, y,w,h)
			f.writelines(text)
	print(indx+1, 'done')

			
			



	f.close()
	