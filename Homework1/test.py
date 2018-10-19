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

x = a_x
y = a_y
x_t = a_x_t
y_t = a_y_t

irls = IRLS(max_step=10000, epsilon=0.0001, modified=False, tau=0.5)
irls.fit(x,y)
irls.plot(x,y)
print(irls.score(x,y))
print(irls.score(x_t,y_t))