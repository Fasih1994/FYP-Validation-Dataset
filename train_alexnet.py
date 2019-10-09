from pyimagesearch.preprocessing import AspectAwarePreprocessor
from pyimagesearch.preprocessing import ImageToArrayPreprocessor
from pyimagesearch.dataset import SimpleDatasetLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
from pyimagesearch.nn.conv import MiniVGGNet
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
from imutils import paths
import argparse
import os

ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dataset', required=True,
                help='Path to dataset')
args = vars(ap.parse_args())

imagePaths = list(paths.list_images(args['dataset']))
print("[INFO] loading Dataset...")
aap = AspectAwarePreprocessor(227, 227)
iap = ImageToArrayPreprocessor()
sdl = SimpleDatasetLoader([aap, iap])
(data, labels) = sdl.load(imagePaths, verbose=500)
le = LabelBinarizer()
labels = le.fit_transform(labels)

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)

print('[INFO] compiling network....')
model = MiniVGGNet.build(227, 227, 3, len(le.classes_))
opt = SGD(lr=0.001)

model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

print('[INFO] training network...')

callback = [ModelCheckpoint('lenet.model', monitor='val_loss',
                            save_best_only=True, verbose=1)]
model.fit(trainX, trainY,
          steps_per_epoch=len(trainX)//64,
          validation_data=(testX, testY),
          batch_size=64, epochs=50,
          callbacks=callback, verbose=1)

print('[INFO] evaluating model...')
preds = model.predict(testX, batch_size=64)
print(classification_report(testY.argmax(axis=1),
                            preds.argmax(axis=1),
                            target_names=le.classes_))