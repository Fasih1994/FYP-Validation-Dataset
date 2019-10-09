from pyimagesearch.preprocessing import AspectAwarePreprocessor
from pyimagesearch.preprocessing import ImageToArrayPreprocessor
from pyimagesearch.callbacks import TrainingMonitor
from keras.preprocessing.image import ImageDataGenerator
from pyimagesearch.dataset import SimpleDatasetLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
from pyimagesearch.nn.conv import AlexNet
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from imutils import paths
import argparse
import os
ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dataset', required=True,
                help='Path to dataset')
ap.add_argument('-e', '--epochs', type=int, default=30,
                help='Number of epochs ')
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

aug = ImageDataGenerator(width_shift_range=0.2,
                         height_shift_range=0.2, brightness_range=[0.2, 1.0],
                         fill_mode='nearest', horizontal_flip=True)

print('[INFO] compiling network....')
model = AlexNet.build(227, 227, 3, len(le.classes_))
opt = Adam(lr=0.001)

model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

print('[INFO] training network...')

callbacks = [ModelCheckpoint('alexnet_with_aug.weights', monitor='val_loss',
                            save_best_only=True, save_weights_only=True, verbose=1),
            TrainingMonitor("{}.png".format(os.getpid()))]
if os.path.exists('alexnet_with_aug.weights'):
    model.load_weights('alexnet_with_aug.weights')
model.fit_generator(aug.flow(trainX, trainY, batch_size=64),
                    steps_per_epoch=len(trainX)//64,
                    epochs=args['epochs'], verbose=1, callbacks=callbacks,
                    validation_data=(testX, testY))

print('[INFO] evaluating model...')
preds = model.predict(testX, batch_size=64)
print(classification_report(testY.argmax(axis=1),
                            preds.argmax(axis=1),
                            target_names=le.classes_))
