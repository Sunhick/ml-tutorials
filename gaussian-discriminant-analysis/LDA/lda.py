# 
# Author: Sunil
# Credit: http://scikit-learn.org/0.16/modules/generated/sklearn.lda.LDA.html
#
import numpy as np
from sklearn.lda import LDA
import matplotlib.pyplot as plt

X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
y = np.array([1, 1, 1, 2, 2, 2])
clf = LDA()

plt.plot([-1, -2, -3], [-1, -1, -2], 'ro')
plt.plot([1, 2, 3], [1, 1, 2], 'bo')
plt.plot([.8], [1], 'wo')
plt.axis([-4, 4, -4, 4])

model = clf.fit(X, y)
print(model.predict([[0.8, 1]]))

plt.show()
#
# useful links: 
#   http://sebastianraschka.com/Articles/2014_python_lda.html
#   http://people.revoledu.com/kardi/tutorial/LDA/index.html
#   https://www.youtube.com/watch?v=azXCzI57Yfc
#