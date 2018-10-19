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
x_t = c_x_t
y_t = c_y_t

d = dict()
d['a'] = load_file('data/classificationA.train'), load_file('data/classificationA.test')
d['b'] = load_file('data/classificationB.train'), load_file('data/classificationB.test')
d['c'] = load_file('data/classificationC.train'), load_file('data/classificationC.test')

models = [LDA, IRLS, LinearRegression, QDA]
for key in d.keys():
    print('Dataset %s :\n' % key)
    (x, y), (x_t, y_t) = d[key]
    for model in models:
        print('\t%s' % model.__name__)
        m = model()
        m.fit(x, y)
        print("\t\tTrain score : %.3f" % m.score(x, y))
        print("\t\tTest score : %.3f" % m.score(x_t, y_t))
        # m.plot(x, y)
    print('\n')
