from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import tensorflow

import matplotlib.pyplot as plt
import numpy as np
import argparse, os

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", required=True, help="Path of DATASET dir")
parser.add_argument("-p", "--plot", type=str, default="ploted.png", help="Path to output loss/acc plot")
parser.add_argument("-m", "--model", type=str,  default="mask_detector.model", help="Path to output human face detector model")
args = vars(parser.parse_args())

# initialize the learning rate, number of epochs to train our model
init_lr= 1e-4 # because we'll apply a learning rate decay schedule
epochs = 20
bs     = 32
lb = LabelBinarizer()

def pre_process():
	
	imgPaths = list(paths.list_images(args["dataset"]))
	data = []
	labels = []

	# walk over all image
	for imgPath in imgPaths:
		label = imgPath.split(os.path.sep)[-2]

		# load image
		image = load_img(imgPath, target_size=(224,224))
		image = img_to_array(image)
		image = preprocess_input(image)

		# update data and label list
		data.append(image)
		labels.append(label)

	# convert data and label to np array
	data = np.array(data, dtype="float32")
	labels = np.array(labels)

	return data, labels

def one_hot(labels):
	# one hot encoging on the labels
	
	labels = lb.fit_transform(labels)
	return to_categorical(labels) # labels

def data_augmentation():
	
	return ImageDataGenerator(
			rotation_range=20,
			zoom_range=0.15,
			width_shift_range=0.2,
			height_shift_range=0.2,
			shear_range=0.15,
			horizontal_flip=True,
			fill_mode="nearest")

def hModel():
	# load the mobile network
	baseModel = MobileNetV2(weights="imagenet", include_top=False, input_tensor=Input(shape=(224,224,3)))

	# construction of the head of model htat will be placed on top of the model
	headModel = baseModel.output
	headModel = AveragePooling2D(pool_size=(7,7))(headModel)
	headModel = Flatten(name="flatten")(headModel)
	headModel = Dense(128, activation="relu")(headModel)
	headModel = Dropout(0.5)(headModel)
	headModel = Dense(2, activation="softmax")(headModel)

	# place the FC model on top
	model = Model(inputs=baseModel.input, outputs=headModel)

	# add the rest o fmodel to the final model
	for layer in baseModel.layers:
		layer.trainable = False

	return model

def compileModel(model):
	opt = Adam(lr=init_lr, decay=init_lr/epochs)
	model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])


def fitModel(model):
	history = model.fit(
		aug.flow(trainX, trainY, batch_size=bs),
		steps_per_epoch=len(trainX) // bs,
		validation_data=(testX, testY),
		validation_steps=len(testX) // bs,
		epochs=epochs)
	print("FIT already done")
	return history

def evaluate():
	pred = model.predict(testX, batch_size=bs)
	pred = np.argmax(pred, axis=1)
	print(classification_report(testY.argmax(axis=1), pred, target_names=lb.classes_))

def saveModel(model):
	model.save(args["model"], save_format="h5")

def History(history):
	n = epochs
	plt.style.use("ggplot")
	plt.figure()
	plt.plot(np.arange(0, n), history.history["loss"], label="train_loss")
	plt.plot(np.arange(0, n), history.history["val_loss"], label="val_loss")
	plt.plot(np.arange(0, n), history.history["accuracy"], label="train_acc")
	plt.plot(np.arange(0, n), history.history["val_accuracy"], label="val_acc")
	plt.title("Training Loss and Accuracy")
	plt.xlabel("Epoch #")
	plt.ylabel("Loss/Acc")
	plt.legend(loc="lower left")
	plt.savefig(args["plot"])


if __name__ == '__main__':
	#tensorflow.debugging.set_log_device_placement(True)
	gpus = tensorflow.config.experimental.list_physical_devices('GPU')
	if len(gpus) == 0:
		gpus = tensorflow.config.experimental.list_physical_devices('CPU')
	try:
		tensorflow.config.experimental.set_virtual_device_configuration(gpus[0],
		[tensorflow.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)])
		print(" - [x]: Loading images ... ->")
		data, labels = pre_process()

		print(" - [x]: One hot on labels ... ->")
		labels = one_hot(labels)

		print(" - [x]: Split dataset ... ->")
		(trainX, testX, trainY, testY) = train_test_split(data, labels, 
			test_size=0.2, stratify=labels, random_state=42)

		print(" - [x]: Data augmentation ... ->")
		aug = data_augmentation()

		print(" - [x]: Prepare the MobileNetV2 for fine-tuning ... ->")
		model = hModel()

		print(" - [x]: Compiling the model ... ->")
		compileModel(model)

		print(" - [x]: Train the model ... ->")
		history = fitModel(model)

		print(" - [x]: evaluate model ... ->")
		evaluate()

		print(" - [x]: Store the model on disk... ->")
		saveModel(model)

		print(" - [x]: Plot the training loss and accuracy ... ->")
		History(history)
	except RuntimeError as e:
		print(e)
	
		
