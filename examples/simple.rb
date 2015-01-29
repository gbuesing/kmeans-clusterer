#!/usr/bin/env ruby

require 'rubygems'
require 'bundler/setup'
require_relative '../lib/kmeans-clusterer'

puts "k-means:\n\n"

data = [
  [3, 3], [-3, 3], [3, -3], [-3, -3],
  [3, 4], [-3, 4], [3, -4], [-3, -4],
  [4, 3], [-4, 3], [4, -3], [-4, -3],
  [4, 4], [-4, 4], [4, -4], [-4, -4],
]

kmeans = KMeansClusterer.run 4, data

kmeans.clusters.each do |cluster|
  puts  cluster.label.to_s + '. ' + # label 1-k
        cluster.centroid.to_s + ": " + 
        cluster.points.join(", ")
end

puts "\nSSE: #{kmeans.error.round(2)}"
puts "Silhouette score: #{kmeans.silhouette_score.round(2)}"

puts "\n---\n"

puts "\nk-medians:\n\n"

kmedians = KMediansClusterer.run 4, data
kmedians.clusters.each do |cluster|
  puts  cluster.label.to_s + '. ' + # label 1-k
        cluster.centroid.to_s + ": " + 
        cluster.points.join(", ")
end

puts "\nSSE: #{kmedians.error.round(2)}"
puts "Silhouette score: #{kmedians.silhouette_score.round(2)}"

# Outputs:
# [3, 3], [3, 4], [4, 3]
# [-3, -3], [-3, -4], [-4, -3]
# [3, -3], [3, -4], [4, -3]
# [-3, 3], [-3, 4], [-4, 3]

puts "\n---\n"

puts "\nCities:\n\n"

latlngs = [[40.71,-74.01], [34.05,-118.24], [39.29,-76.61], 
          [45.52,-122.68], [38.9,-77.04], [36.11,-115.17]]

labels = ['New York', 'Los Angeles', 'Baltimore', 
          'Portland', 'Washington DC', 'Las Vegas']

kmeans = KMeansClusterer.run 2, latlngs, labels: labels, runs: 1

kmeans.clusters.each do |cluster|
  puts  cluster.label.to_s + '. ' + 
        cluster.points.map(&:label).join(", ") + "\t" +
        cluster.centroid.to_a.map {|v| v.round(2)}.to_s
end

# Use existing clusters for prediction with new data:
cluster = kmeans.closest_cluster [41.85,-87.65] # Chicago
puts "\nClosest cluster to Chicago: #{cluster.label}"

puts "\nSSE: #{kmeans.error.round(2)}"
puts "Silhouette score: #{kmeans.silhouette_score.round(2)}"




