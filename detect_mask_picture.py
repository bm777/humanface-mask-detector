from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import tensorflow
import numpy as np
import argparse
import cv2
import os

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--image", required=True, help="Path to image")
parser.add_argument("-f", "--face", type=str, default="simple_face_detector")
parser.add_argument("-m", "--model", type=str, default="mask_detector.model",
	help="Path to trained face mask detector model")
parser.add_argument("-c", "--confidence", type=float, default=0.5,
	help="Minimum probability to filter weak detection")
args = vars(parser.parse_args())

def load():
	proto = os.path.sep.join([args["face"], "deploy.prototxt"])
	weights = os.path.sep.join([args["face"], "res10_300x300_ssd_iter_140000.caffemodel"])
	net = cv2.dnn.readNet(proto, weights)

	model = load_model(args["model"])
	return model, net

def process(model, net):
	image = cv2.imread(args["image"])
	origne = image.copy()
	(h, w) = image.shape[:2]

	blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),
		(104.0, 177.0, 123.0))
	net.setInput(blob)
	detections = net.forward()
	for i in range(0, detections.shape[2]):
		confidence = detections[0, 0, i, 2]
		if confidence > args["confidence"]:
			box = detections[0, 0, i, 3:7] * np.array([w,h,w,h])
			(startX, startY, endX, endY) = box.astype("int")

			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w-1, endX), min(h-1, endY))


			# extract face ROI, convert it to RGB channel
			face = image[startY:endY, startX:endX]
			face = cv2.flip(face, 1)
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224,224))
			face = img_to_array(face)
			face = preprocess_input(face)
			face = np.expand_dims(face, axis=0)

			#pass the face through the model to determince if the face has a mask or not
			(mask, withoutMask) = model.predict(face)[0]

			label = "Mask" if mask > withoutMask else "No Mask"
			color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

			label = "{}: {:.2f}%".format(label, max(mask, withoutMask)*100)

			cv2.putText(image, label, (startX, startY-10),
				cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
			cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)
	cv2.imshow("Ouput", image)

	if cv2.waitKey(0) == ord('Q'):
		cv2.destroyAllWindows()



if __name__ == '__main__':

	gpus = tensorflow.config.experimental.list_physical_devices('GPU')
	print("------------"+str(len(tensorflow.config.experimental.list_physical_devices('CPU'))))
	if len(gpus) == 0:
		gpu = tensorflow.config.experimental.list_physical_devices('CPU')
	try:
		tensorflow.config.experimental.set_virtual_device_configuration(gpus[0],
		[tensorflow.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)])

		print(" - [x]: Load the model from disk... ->")
		model, net = load()

		print(" - [x]: Trying to predict ... ->")
		process(model,  net)
	except Exception as e:
		raise e
