from models import HMM
import numpy as np
import warnings

warnings.filterwarnings("ignore")


def load_file(file_name):
    '''
    Convert the file to an array
    :param file_name: path of the file, each line of the file is of the form x y
    :type file_name: str
    :return: Array of the coordinates described by the file
    :rtype: numpy.array
    '''

    def convert(line):
        x1, x2 = line.split()
        return np.array([np.float(x1), np.float(x2)])

    with open(file_name) as f:
        content = np.array([convert(line) for line in f])
    return np.array(content)


mus = np.array([[-2.03436695, 4.17258596],
                [3.80070949, -3.79729742],
                [-3.06196072, -3.53454048],
                [3.97793025, 3.77333059]])

sigmas = np.array([[[2.90442381, 0.20655763],
                    [0.20655763, 2.75617077]],
                   [[0.92127927, 0.0573808],
                    [0.0573808, 1.86586017]],
                   [[6.24140909, 6.05017464],
                    [6.05017464, 6.18245528]],
                   [[0.21035667, 0.29045085],
                    [0.29045085, 12.23996609]]])

pi = np.array([0.25155719, 0.18290156, 0.30555552, 0.25998574])

data = load_file('data/EMGaussian.data')
data_test = load_file('data/EMGaussian.test')

A = np.full((4, 4), 1 / 4)

hmm = HMM(A, 4, pi, mus, sigmas, data)

hmm.em(100)

labels = hmm.viterbi()
labels_t = hmm.viterbi(data_test)
