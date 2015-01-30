#!/usr/bin/env ruby

require 'bundler/setup'
require_relative '../lib/kmeans-clusterer'
require 'optparse'

k = 10
runs = 10
d = 10 # point dimension
n = 1000 # points per cluster
o = 10 # cluster offset

OptionParser.new do |opts|
  opts.on("-kK") {|v| k = v.to_i }
  opts.on("-dD") {|v| d = v.to_i }
  opts.on("-nN") {|v| n = v.to_i }
  opts.on("-oO") {|v| o = v.to_i }
end.parse!

data = k.times.map do
  cluster = NArray.float(d,n).randomn
  offset = NArray.int(d).random(o) - o/2
  (cluster + offset).to_a
end.reduce(:+)


puts "Clustering #{data.length} #{d}-D points into #{k} clusters...\n\n"

t = Time.now
kmeans = KMeansClusterer.run k, data, runs: runs, log: true
elapsed = Time.now - t

puts "\nBest of #{runs} runs (total time #{elapsed.round(2)}s):"
puts "#{k} clusters in #{kmeans.iterations} iterations, #{kmeans.runtime.round(2)}s, SSE #{kmeans.error.round(2)}"
puts "Silhouette score: #{kmeans.silhouette_score.round(2)}"
