#!/usr/bin/env ruby

require 'rubygems'
require 'bundler/setup'
require_relative '../lib/kmeans-clusterer'

puts "Basic 2D:\n\n"

data = [
  [3, 3], [-3, 3], [3, -3], [-3, -3],
  [3, 4], [-3, 4], [3, -4], [-3, -4],
  [4, 3], [-4, 3], [4, -3], [-4, -3],
  [4, 4], [-4, 4], [4, -4], [-4, -4],
]

result = KMeansClusterer.run 4, data

result.clusters.each do |cluster|
  puts  cluster.label.to_s + '. ' + # label 1-k
        cluster.center.to_s + ": " + 
        cluster.points.join(", ")
end

puts "\nSSE: #{result.sum_of_squares_error.round(2)}"
puts "Silhouette score: #{result.silhouette_score.round(2)}"

# Outputs:
# [3, 3], [3, 4], [4, 3]
# [-3, -3], [-3, -4], [-4, -3]
# [3, -3], [3, -4], [4, -3]
# [-3, 3], [-3, 4], [-4, 3]

puts "\nCities:\n\n"

data = [[40.71,-74.01], [34.05,-118.24], [39.29,-76.61], 
        [45.52,-122.68], [38.9,-77.04], [36.11,-115.17]]

labels = ['New York', 'Los Angeles', 'Baltimore', 
          'Portland', 'Washington DC', 'Las Vegas']

result = KMeansClusterer.run 2, data, labels: labels, runs: 1

result.clusters.each do |cluster|
  puts  cluster.label.to_s + '. ' + 
        cluster.points.map(&:label).join(", ")
end

puts "\nSSE: #{result.sum_of_squares_error.round(2)}"
puts "Silhouette score: #{result.silhouette_score.round(2)}"