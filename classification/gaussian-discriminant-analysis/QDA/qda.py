from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import numpy as np
import matplotlib.pyplot as plt

X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
y = np.array([1, 1, 1, 2, 2, 2])

plt.plot([-1, -2, -3], [-1, -1, -2], 'ro')
plt.plot([1, 2, 3], [1, 1, 2], 'bo')
plt.plot([.8], [1], 'wo')
plt.axis([-4, 4, -4, 4])

clf = QuadraticDiscriminantAnalysis()
clf.fit(X, y)

print(clf.predict([[-0.8, -1]]))
plt.show()