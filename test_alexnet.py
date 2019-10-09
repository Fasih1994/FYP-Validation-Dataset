from pyimagesearch.nn.conv import AlexNet
from pyimagesearch.preprocessing import SimplePreprocessor, ImageToArrayPreprocessor
from pyimagesearch.dataset import SimpleDatasetLoader
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from imutils import paths
import argparse
import random

ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dataset', required=True,
                help="path to input dataset")
ap.add_argument('-w', '--weights', required=True,
                help='path to network weights')
args = vars(ap.parse_args())

imagePaths = paths.list_images(args['dataset'])
random.shuffle(imagePaths)
sp = SimplePreprocessor(227, 227)
iap = ImageToArrayPreprocessor()
sdl = SimpleDatasetLoader([sp, iap])
(data, labels) = sdl.load(imagePaths, verbose=500)
model = AlexNet.build()
model.load_weights(args['weights'])
le= LabelEncoder()
labels = le.fit_transform(labels)
print('[INFO] predicting....')
pred = model.predict(data)

print(classification_report(labels, pred.argmax(axis=1),
                            target_names=le.classes_))