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




x, y = load_file('data/classificationA.test')



x_t, y_t = load_file('data/classificationA.train')

lr = LinearRegression()
lr.fit(x,y)
print(lr.score(x_t,y_t))
lr.plot(x_t,y_t)
# print(lda.score(x_t, y_t))

