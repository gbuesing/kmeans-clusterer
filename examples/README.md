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

2. run ```./examples/mnist.rb```

After running k-means, a test set of digits will be classified (by finding the closest cluster) and outputted to a PNG with each cluster represented as a row.

Example PNG output:

![MNIST clustering example](https://raw.githubusercontent.com/gbuesing/kmeans-clusterer/master/examples/data/mnist_example.png)
