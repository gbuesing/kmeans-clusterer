#!/usr/bin/env ruby

require 'rubygems'
require 'bundler/setup'
require_relative '../lib/kmeans-clusterer'
require 'csv'

k = (ARGV[0] || 20).to_i
runs = (ARGV[1] || 10).to_i

data, labels = [], []
CSV.foreach("examples/data/us_cities.csv") do |row|
  data << [row[2].to_f, row[3].to_f]
  labels << "#{row[1]} #{row[0]}"
end

t = Time.now
kmeans = KMeansClusterer.run(k, data, labels: labels, runs: runs)
elapsed = Time.now - t

kmeans.sorted_clusters.each do |cluster|
  puts "\n#---\n\n"
  cluster.sorted_points.each do |point|
    puts point.label
  end
end

puts "\nBest of #{runs} runs (total time #{elapsed.round(2)}s):"
puts "#{k} clusters in #{kmeans.iterations} iterations, #{kmeans.runtime.round(2)}s, SSE #{kmeans.sum_of_squares_error.round(2)}"
puts "Silhouette score: #{kmeans.silhouette_score.round(2)}"
