KMeansClusterer
===

[k-means clustering](http://en.wikipedia.org/wiki/K-means_clustering) in Ruby. Uses [NArray](https://github.com/masa16/narray) under the hood for fast calculations.

Jump to the [examples](examples/) directory to see this in action.


Features
---

- Runs multiple clustering attempts to find optimal solution (single runs are susceptible to falling into non-optimal local minima)
- Initializes centroids via [k-means++](http://en.wikipedia.org/wiki/K-means%2B%2B) algorithm, for faster convergence
- Calculates [silhouette](http://en.wikipedia.org/wiki/Silhouette_%28clustering%29) score for evaluation
- Option to scale data before clustering, so that output isn't biased by different feature scales
- Works with high-dimensional data


Install
---
```gem install kmeans-clusterer```


Usage
---

Simple example:

```ruby
require 'kmeans-clusterer'

data = [[40.71,-74.01],[34.05,-118.24],[39.29,-76.61],
        [45.52,-122.68],[38.9,-77.04],[36.11,-115.17]]

labels = ['New York', 'Los Angeles', 'Baltimore', 
          'Portland', 'Washington DC', 'Las Vegas']

k = 2 # find 2 clusters in data

kmeans = KMeansClusterer.run k, data, labels: labels, runs: 5

kmeans.clusters.each do |cluster|
  puts  cluster.id.to_s + '. ' + 
        cluster.points.map(&:label).join(", ") + "\t" +
        cluster.centroid.to_s
end

# Use existing clusters for prediction with new data:
predicted = kmeans.predict [[41.85,-87.65]] # Chicago
puts "\nClosest cluster to Chicago: #{predicted[0]}"

# Clustering quality score. Value between -1.0..1.0 (1.0 is best)
puts "\nSilhouette score: #{kmeans.silhouette.round(2)}"
```

Output of simple example:

```
0. New York, Baltimore, Washington DC [39.63, -75.89]
1. Los Angeles, Portland, Las Vegas [38.56, -118.7]

Closest cluster to Chicago: 0

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
:float_precision | :double | float precision to use. :double or :single

