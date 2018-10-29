import numpy as np


def load_file(file_name):
    def convert(line):
        x1, x2 = line.split()
        return np.array([np.float(x1), np.float(x2)])

    with open(file_name) as f:
        content = np.array([convert(line) for line in f])
    return content
