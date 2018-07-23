import numpy as np
import utils


class Quantize:

    def __init__(self, b):
        self.k = 2**b
        #print(self.k)
    def quantize(self, I):
        #print(I)
        N, D = I.shape
        y = np.ones(N)

        means = np.zeros((self.k, D))
        for kk in range(self.k):
            i = np.random.randint(N)
            means[kk] = I[i]

        while True:
            y_old = y

            # Compute euclidean distance to each mean
            dist2 = utils.euclidean_dist_squared(I, means)
            dist2[np.isnan(dist2)] = np.inf
            y = np.argmin(dist2, axis=1)

            # Update means
            for kk in range(self.k):
                means[kk] = I[y==kk].mean(axis=0)

            changes = np.sum(y != y_old)
            # print('Running K-means, changes in cluster assignment = {}'.format(changes))

            # Stop if no point changed cluster
            if changes == 0:
                break

        self.means = means
        self.y = y
        #print(means)

    def dequantize(self, I):
        y = self.y
        means = self.means
        print(np.shape(I))
        for i in range(len(y)):
            I[i-1] = means[y[i-1]]

    # def error(self, I):
        # error = 0
        # means = self.means
        # y = self.y
        # for i in range(len(y)):
            # for n in range(len(I[i])):
                # error += np.square(I[i][n]-means[y[i]][n])
        
# return error