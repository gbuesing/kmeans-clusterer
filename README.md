KMeansClusterer
===

[k-means clustering](http://en.wikipedia.org/wiki/K-means_clustering) in Ruby. Uses [NArray](https://github.com/masa16/narray) under the hood for fast calculations.

Runs multiple clustering attempts with different initial centerpoints and returns run with lowest error. This helps ensure that k-means will return optimal clustering for k, vs. getting stuck in a local minimum.


Usage
---
Simple example:

```ruby
data = [[40.71,-74.01],[34.05,-118.24],[39.29,-76.61],
        [45.52,-122.68],[38.9,-77.04],[36.11,-115.17]]

labels = ['New York', 'Los Angeles', 'Baltimore', 
          'Portland', 'Washington DC', 'Las Vegas']

k = 2 # find 2 clusters in data

# Options:
#   labels: array of Ruby objects to collate with data array
#   runs: number of times to run kmeans (default is 10)
kmeans = KMeansClusterer.run k, data, labels: labels, runs: 3

kmeans.clusters.each do |cluster|
  puts  cluster.label.to_s + '. ' + 
        cluster.points.map(&:label).join(", ")
end

puts "\nSilhouette score: #{kmeans.silhouette_score.round(2)}"

# Outputs:
#
# 1. Los Angeles, Portland, Las Vegas
# 2. New York, Baltimore, Washington DC
#
# Silhouette score: 0.91
```


TODO
---
- more distance measures
- mini batch
- kmeans++ initialization
- more examples
