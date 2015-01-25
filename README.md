KMeansClusterer
===

[KMeans clustering](http://en.wikipedia.org/wiki/K-means_clustering) in Ruby. Uses [NArray](https://github.com/masa16/narray) under the hood for fast calculations.

Runs multiple clusterings and returns run with lowest error, to avoid converging on local minimum.


Usage
---
Simple example:

```ruby
data = [
  [3, 3], [-3, 3], [3, -3], [-3, -3],
  [3, 4], [-3, 4], [3, -4], [-3, -4],
  [4, 3], [-4, 3], [4, -3], [-4, -3],
]

kmeans = KMeansClusterer.run 4, data

kmeans.clusters.each do |cluster|
  puts cluster.points.join(', ')
end

# Outputs:
# [3, 3], [3, 4], [4, 3]
# [-3, -3], [-3, -4], [-4, -3]
# [3, -3], [3, -4], [4, -3]
# [-3, 3], [-3, 4], [-4, 3]
```


TODO
---
- more distance measures
- mini batch
- kmeans++ initialization
- more examples
