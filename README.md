KMeansClusterer
===

[k-means](http://en.wikipedia.org/wiki/K-means_clustering) and [k-medians](http://en.wikipedia.org/wiki/K-medians_clustering) clustering in Ruby. Uses [NArray](https://github.com/masa16/narray) under the hood for fast calculations.

- Runs multiple clustering attempts to find optimal solution (single runs are susceptible to falling into non-optimal local minima)
- Initializes centroids via [k-means++](http://en.wikipedia.org/wiki/K-means%2B%2B) algorithm, for faster convergence
- Calculates [silhouette](http://en.wikipedia.org/wiki/Silhouette_%28clustering%29) score for evaluation
- Option to scale data before clustering, so that output isn't biased by different feature scales
- Works with high-dimensional data


Usage
---

Simple example:

```ruby
data = [[40.71,-74.01],[34.05,-118.24],[39.29,-76.61],
        [45.52,-122.68],[38.9,-77.04],[36.11,-115.17]]

labels = ['New York', 'Los Angeles', 'Baltimore', 
          'Portland', 'Washington DC', 'Las Vegas']

k = 2 # find 2 clusters in data

kmeans = KMeansClusterer.run k, data, labels: labels, runs: 5

kmeans.clusters.each do |cluster|
  puts  cluster.label.to_s + '. ' + 
        cluster.points.map(&:label).join(", ") + "\t" +
        cluster.centroid.to_s
end

# Use existing clusters for prediction with new data:
cluster = kmeans.closest_cluster [41.85,-87.65] # Chicago
puts "\nClosest cluster to Chicago: #{cluster.label}"

# Clustering quality score. Value between -1.0..1.0 (1.0 is best)
puts "\nSilhouette score: #{kmeans.silhouette_score.round(2)}"
```

Output of simple example:

```
1. New York, Baltimore, Washington DC [39.63, -75.89]
2. Los Angeles, Portland, Las Vegas [38.56, -118.7]

Closest cluster to Chicago: 1

Silhouette score: 0.91
```

### Options

The following options can be passed in to ```KMeansClusterer.run```:

option | default | description
------ | ------- | -----------
:labels | nil | optional array of Ruby objects to collate with data array
:runs   | 10 | number of times to run kmeans
:log    | false | Print stats after each run
:init   | :kmpp | algorithm for picking initial cluster centroids. Accepts :kmpp, :random, or an array of k centroids
:scale_data | false | Scales features to -1..1 range

### KMediansClusterer

k-medians clustering is available via ```KMediansClusterer```, which has the same api
as ```KMeansClusterer```:

```ruby
kmedians = KMediansClusterer.run k, data, labels: labels, runs: 5
```

Differences between the two variants:

attribute | KMeansClusterer | KMediansClusterer
- | --------------- | -----------------
distance measure | Euclidean | Manhattan
centroid calculation | mean | median
cost function | sum of squared distances | sum of distances



### More examples

More usage examples are available in the [examples](examples/) directory.


License
---
MIT
