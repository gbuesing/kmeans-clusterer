#!/usr/bin/env ruby

require 'rubygems'
require 'bundler/setup'
require_relative '../lib/kmeans-clusterer'
require 'csv'

k = (ARGV[0] || 20).to_i
runs = (ARGV[1] || 10).to_i

points, tags = [], []
CSV.foreach("examples/us_cities.csv") do |row|
  points << [row[2].to_f, row[3].to_f]
  tags << "#{row[1]} #{row[0]}"
end

t = Time.now
kmeans = KMeansClusterer.run(k, points, tags: tags, runs: runs)
elapsed = Time.now - t

kmeans.sorted_clusters.each do |cluster|
  puts "\n---\n\n"
  cluster.sorted_points.each do |point|
    puts point.tag
  end
end

puts "\nBest of #{runs} runs (total time #{elapsed.round(2)}s):"
puts "#{k} clusters in #{kmeans.iterations} iterations, #{kmeans.runtime.round(2)}s, SSE #{kmeans.sum_of_squares_error.round(2)}"
