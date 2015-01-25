#!/usr/bin/env ruby

require 'rubygems'
require 'bundler/setup'
require_relative '../lib/kmeans-clusterer'


data = [
  [3, 3], [-3, 3], [3, -3], [-3, -3],
  [3, 4], [-3, 4], [3, -4], [-3, -4],
  [4, 3], [-4, 3], [4, -3], [-4, -3],
]

kmeans = KMeansClusterer.run 4, data

kmeans.clusters.each do |cluster|
  puts cluster.points.join(', ')
end

puts "\nSSE #{kmeans.sum_of_squares_error.round(2)}, silhouette score: #{kmeans.silhouette_score.round(2)}"

# Outputs:
# [3, 3], [3, 4], [4, 3]
# [-3, -3], [-3, -4], [-4, -3]
# [3, -3], [3, -4], [4, -3]
# [-3, 3], [-3, 4], [-4, 3]
