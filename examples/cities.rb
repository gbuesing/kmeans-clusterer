#!/usr/bin/env ruby

require 'rubygems'
require 'bundler/setup'
require_relative '../lib/kmeans-clusterer'
require 'csv'
require 'optparse'

k = 10
runs = 10
skip_plot = false

OptionParser.new do |opts|
  opts.on("-kK") {|v| k = v.to_i }
  opts.on("--skip-plot") {|v| skip_plot = true }
end.parse!

cities = CSV.foreach("examples/data/us_cities.csv").map do |row|
  { name: row[0], state: row[1], lat: row[2].to_f, lng: row[3].to_f }
end

data = cities.map {|city| [city[:lat], city[:lng]] }

t = Time.now
kmeans = KMeansClusterer.run(k, data, labels: cities, runs: runs, scale_data: true)
elapsed = Time.now - t

kmeans.sorted_clusters.each do |cluster|
  puts "\n#---\n\n"
  cluster.points.each do |point|
    city = point.label
    puts "#{city[:state]} #{city[:name]}"
  end
end

puts "\nBest of #{runs} runs (total time #{elapsed.round(2)}s):"
puts "#{k} clusters in #{kmeans.iterations} iterations, #{kmeans.runtime.round(2)}s, SSE #{kmeans.error.round(2)}"
puts "Silhouette score: #{kmeans.silhouette_score.round(2)}"

unless skip_plot
  require 'gnuplot'

  outfile = "examples/data/output/cities_k#{k}.png"

  Gnuplot.open do |gp|
    Gnuplot::Plot.new(gp) do |plot|
      plot.terminal "png"
      plot.output outfile

      kmeans.clusters.each do |cluster|
        x = cluster.points.map { |p| p.label[:lng] }
        y = cluster.points.map { |p| p.label[:lat] }

        plot.data << Gnuplot::DataSet.new([x, y]) do |ds|
          ds.notitle
        end
      end
    end
  end

  `open #{outfile}`
end
