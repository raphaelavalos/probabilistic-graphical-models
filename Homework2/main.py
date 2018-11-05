import numpy as np
from models import *

x = load_file('data/EMGaussian.data')
x_test = load_file('data/EMGaussian.test')

print("Running kmean++ 5 times")
for i in range(5):
    print("\n\t- Run %i" % (i + 1))
    kmean = KMeans()
    dist = kmean.fit(x, return_distortion=True)
    print('\t\tDist: %.5f' % dist)
    print('\t\tCentroids: ', kmean.cluster_centers_)
    kmean.plot(x)
    plt.show()

print("\nRunning EM\n")
print("\t- EM ISO\n")
em = EM()
em.fit(x, iso=True)
em.plot()
_, logl_train = em.predict(x)
_, logl_test = em.predict(x_test)
print('\t\t Log likelihood data: %.5f' % logl_train)
print('\t\t Log likelihood test: %.5f' % logl_test)

print("\n\t- EM General\n")
em = EM()
em.fit(x, iso=False)
em.plot()
_, logl_train = em.predict(x)
_, logl_test = em.predict(x_test)
print('\t\t Log likelihood data: %.5f' % logl_train)
print('\t\t Log likelihood test: %.5f' % logl_test)
