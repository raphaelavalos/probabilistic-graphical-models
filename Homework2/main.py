import numpy as np
from models import *

x = load_file('data/EMGaussian.data')
x_test = load_file('data/EMGaussian.test')
for i in range(0):
    kmean = KMeans()
    dist = kmean.fit(x, return_distortion=True)
    print('Dist', dist)
    print('centroids ', kmean.cluster_centers_)
    kmean.plot(x)
    plt.show()

em = EM()
em.fit(x,sphere=False)
em.plot()
print(em.mus)
_, logl_train = em.predict(x)
_, logl_test = em.predict(x_test)