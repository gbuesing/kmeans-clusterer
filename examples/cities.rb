#!/usr/bin/env ruby

require 'rubygems'
require 'bundler/setup'
require_relative '../lib/kmeans-clusterer'
require 'csv'

k = (ARGV[0] || 20).to_i

points, tags = [], []
CSV.foreach("examples/us_cities.csv") do |row|
  points << [row[2].to_f, row[3].to_f]
  tags << "#{row[1]} #{row[0]}"
end

kmeans = KMeansClusterer.new(k, points, tags: tags)
run = kmeans.run

kmeans.sorted_clusters.each do |cluster|
  puts "\n---\n\n"
  cluster.sorted_points.each do |point|
    puts point.tag
  end
end

puts "\n#{k} clusters in #{run[:iterations]} iterations, #{run[:time].round(2)}s, SSE #{kmeans.sum_of_squares_error.round(2)}"