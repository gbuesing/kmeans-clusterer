#!/usr/bin/env ruby

require 'rubygems'
require 'bundler/setup'
require_relative '../lib/kmeans-clusterer'
require_relative './utils/bag'
require 'optparse'

# Data from Qazvinian and radev 2011 http://www-personal.umich.edu/~vahed/data.html
datafiles = Dir['examples/data/headlines/*.txt']
basenames = datafiles.map {|f| File.basename(f, '.txt')}

k = datafiles.length
runs = 10

OptionParser.new do |opts|
  opts.on("-kK") {|v| k = v.to_i }
  opts.on("-rD") {|v| runs = v.to_i }
end.parse!


docs = []
doc_fileids = []

get_basename = -> (docid) {
  fileid = doc_fileids[docid]
  basenames[fileid]
}

bag = BagOfWords.new idf: true

datafiles.each_with_index do |filename, i|
  File.open(filename).each do |line|
    doc = line.chomp.to_s
    bag.add_doc doc
    docs << doc
    doc_fileids << i
  end
end

puts "\nClassifying #{docs.length} docs with #{bag.terms_count} unique terms into #{k} clusters:\n"


start = Time.now
kmeans = KMeansClusterer.run(k, bag.to_a, runs: runs, log: true)
elapsed = Time.now - start

kmeans.clusters.each do |cluster|
  puts "\nc#{cluster.id} - #{cluster.points.length} docs\n"

  acc = Hash.new {|h, k| h[k] = []}
  grouped_points = cluster.points.inject(acc){|hsh, p| hsh[get_basename[p.id]] << p; hsh }
  sums = grouped_points.map {|file, points| [file, points.length]}
  puts sums.map {|(k, v)| "#{k}: #{v}"}.join(', ')

  samplesize = 5
  samplesize = 2 if grouped_points.keys.length > 2
  samplesize = 1 if grouped_points.keys.length > 10

  grouped_points.each do |name, points|
    points.sample(samplesize).each do |point|
      puts "[#{name}] #{docs[point.id]}"
    end
  end
end

puts "\nBest of #{runs} runs (total time #{elapsed.round(2)}s):"
puts "#{k} clusters in #{kmeans.iterations} iterations, #{kmeans.runtime.round(2)}s, SSE #{kmeans.error.round(2)}"
