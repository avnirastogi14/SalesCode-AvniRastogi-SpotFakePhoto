import numpy as np
from fusion.classifier import ScreenClassifier
X = np.random.rand(20, 50)
y = np.random.randint(0, 2, 20)
clf = ScreenClassifier()
clf.train(X, y)
print(clf.predict(X[0]))