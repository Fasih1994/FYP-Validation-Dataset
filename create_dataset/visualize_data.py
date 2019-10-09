import os
import random
from imutils import paths

dataPath = 'data'
imagePaths = list(paths.list_images(dataPath))
random.shuffle(imagePaths)
letters = os.listdir(dataPath+'/Ahmed')
labels = [l.split(os.path.sep)[-1].split('.')[0].split('_')[0] for l in imagePaths]

for letter in sorted(letters):
    print("{} counts = {}".format(letter, labels.count(letter)))

print('[INFO] total counts:{}'.format(len(labels)))