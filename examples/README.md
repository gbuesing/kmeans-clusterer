KMeansClusterer Examples
===


US Cities
---

This example clusters US cities based on lat/lng and outputs the clusters to the terminal and to a PNG (requires GNUPlot.)

The number of clusters can be configured on the command line:

```./examples/cities.rb -k 10```

![Cities clustering example](https://raw.githubusercontent.com/gbuesing/kmeans-clusterer/master/examples/data/cities_k10.png)


Pick Best Value for k
---

This example shows how to pick the best value for k using both the elbow method and the silhouette method.

```./examples/pick_k.rb``` # requires GNUPlot

Initial setup of points, with 4 fairly well-defined clusters:

![unclustered points](https://raw.githubusercontent.com/gbuesing/kmeans-clusterer/master/examples/data/unclustered.png)

Elbow method - find the point of diminishing returns:

![chart of elbow for k](https://raw.githubusercontent.com/gbuesing/kmeans-clusterer/master/examples/data/elbow.png)

Silhouette method - pick k with the highest silhouette score

![chart of silhouette for k](https://raw.githubusercontent.com/gbuesing/kmeans-clusterer/master/examples/data/silhouette.png)

Points plotted with best k value of 4:

![plot of points with best k](https://raw.githubusercontent.com/gbuesing/kmeans-clusterer/master/examples/data/best_k.png)


MNIST Handwritten Digits
---

This example clusters handwritten digits from the [MNIST database of handwritten digits](http://yann.lecun.com/exdb/mnist/).

To run this example:

1. download the MNIST [training set images](http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz) and [training set labels](http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz) and place them in ```examples/data/mnist/```

2. run ```./examples/mnist.rb -k 10```

After running k-means, a test set of digits will be classified (by finding the closest cluster) and outputted to a PNG with each cluster represented as a row.

Example PNG output with k=20:

![MNIST clustering example](https://raw.githubusercontent.com/gbuesing/kmeans-clusterer/master/examples/data/mnist_k20.png)
