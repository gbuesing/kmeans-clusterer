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
  [4, 4], [-4, 4], [4, -4], [-4, -4],
]

result = KMeansClusterer.run 4, data

result.clusters.each do |cluster|
  puts cluster.points.join(', ')
end

puts "\nSSE: #{result.sum_of_squares_error.round(2)}"
puts "Silhouette score: #{result.silhouette_score.round(2)}"

# Outputs:
#
# [3, 3], [3, 4], [4, 3], [4, 4]
# [-3, -3], [-3, -4], [-4, -3], [-4, -4]
# [3, -3], [3, -4], [4, -3], [4, -4]
# [-3, 3], [-3, 4], [-4, 3], [-4, 4]
#
# SSE: 8.0
# Silhouette score: 0.87
```


TODO
---
- more distance measures
- mini batch
- kmeans++ initialization
- more examples
