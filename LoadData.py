import numpy as np
import pickle
import scipy.misc
import os
import matplotlib.pyplot as plt
xs = []
ys = []

DATA_FOLDER = 'driving_dataset'
TRAIN_FILE = os.path.join(DATA_FOLDER, 'data.txt')

with open(TRAIN_FILE) as f:
    for line in f:
        path, angle = line.strip().split()
        xs.append(os.path.join(DATA_FOLDER, path))
        ys.append(float(angle) * scipy.pi / 180)

features = []
for i in (range(len(xs))):
    img = scipy.misc.imread(xs[i])[-150:]
    features.append(scipy.misc.imresize(img, [66, 200])/ 255.0)


features = np.array(features).astype('float32')
labels = np.array(ys).astype('float32')

with open("features", "wb") as f:
    pickle.dump(features, f, protocol=4)
with open("labels", "wb") as f:
    pickle.dump(labels, f, protocol=4)
