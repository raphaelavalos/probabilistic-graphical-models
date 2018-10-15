import numpy as np
import matplotlib.pyplot as plt

from models import *


def load_file(file_name):
    def convert(line):
        x1, x2, y = line.split()
        return np.array([np.float(x1), np.float(x2)]), np.int(np.float(y))

    with open(file_name) as f:
        content = np.array([convert(line) for line in f])
    x, y = zip(*content)
    return np.array(x), np.array(y)


a_x, a_y = load_file('data/classificationA.train')
b_x, b_y = load_file('data/classificationB.train')
c_x, c_y = load_file('data/classificationC.train')
a_x_t, a_y_t = load_file('data/classificationA.test')
b_x_t, b_y_t = load_file('data/classificationB.test')
c_x_t, c_y_t = load_file('data/classificationC.test')

x = c_x
y = c_y
x_t = c_x_t
y_t = c_y_t


lda = LDA()
irls = IRLS()
lr = LinearRegression()
qda = QDA()

models = [lda,irls,lr,qda]
for m in models :
    print(m.__class__.__name__)
    m.fit(x,y)
    m.plot(x,y)
    print("Train score : %.3f" % m.score(x,y))
    print("Test score : %.3f" % m.score(x_t,y_t))
