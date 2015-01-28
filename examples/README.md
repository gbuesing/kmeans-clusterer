KMeansClusterer Examples
===


US Cities
---

This example clusters US cities based on lat/lng and outputs the clusters to the terminal.

The number of clusters can be configured on the command line:

```./examples/cities.rb '12'```


MNIST
---

This example clusters handwritten digits from the [MNIST database of handwritten digits](http://yann.lecun.com/exdb/mnist/).

To run this example:

1. download the MNIST [training set images](http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz) and [training set labels](http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz) and place them in ```examples/data/mnist/```

2. run ```./examples/mnist.rb '10' # k=10```

After running k-means, a test set of digits will be classified (by finding the closest cluster) and outputted to a PNG with each cluster represented as a row.

Example PNG output with k=20:

![MNIST clustering example](https://raw.githubusercontent.com/gbuesing/kmeans-clusterer/master/examples/data/mnist_k20.png)
