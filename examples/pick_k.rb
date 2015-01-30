#!/usr/bin/env ruby

require 'rubygems'
require 'bundler/setup'
require_relative '../lib/kmeans-clusterer'
require 'optparse'

skip_plot = false

OptionParser.new do |opts|
  opts.on("--skip-plot") {|v| skip_plot = true }
end.parse!


# create 4 constellations of data
data1 = (NArray.float(2,200).randomn + NArray[3,5]).to_a
data2 = (NArray.float(2,200).randomn + NArray[-5,4]).to_a
data3 = (NArray.float(2,200).randomn + NArray[-6,-4]).to_a
data4 = (NArray.float(2,200).randomn + NArray[2,-1]).to_a

data = data1 + data2 + data3 + data4

unless skip_plot
  require 'gnuplot'

  unclustered = "examples/data/output/unclustered.png"

  Gnuplot.open do |gp|
    Gnuplot::Plot.new(gp) do |plot|
      plot.terminal "png"
      plot.output unclustered

      x = data.map { |p| p[0] }
      y = data.map { |p| p[1] }

      plot.data << Gnuplot::DataSet.new([x, y]) do |ds|
        ds.notitle
        ds.linecolor = 0
      end
    end
  end

  `open #{unclustered}`
end


ks = 2.upto(10).to_a
errors, silhouettes = [], []

puts "k\tsilhouette\tsse"
runs = ks.map do |k|
  kmeans = KMeansClusterer.run k, data, runs: 3
  error, ss = kmeans.error, kmeans.silhouette_score
  errors << error
  silhouettes << ss
  puts "#{k}\t#{ss.round(2)}\t\t#{error.round(1)}"
  kmeans
end


index_of_max_ss = silhouettes.index silhouettes.max
bestrun = runs[index_of_max_ss]

puts "\nBest choice of k=#{bestrun.k} with silhouette=#{silhouettes[index_of_max_ss].round(2)}"

unless skip_plot
  elbowfile = "examples/data/output/elbow.png"

  Gnuplot.open do |gp|
    Gnuplot::Plot.new( gp ) do |plot|
      plot.terminal "png"
      plot.output elbowfile
      plot.xlabel "k"
      plot.ylabel "SSE"
      
      plot.data << Gnuplot::DataSet.new([ks, errors]) do |ds|
        ds.with = "lines"
        ds.notitle
      end
    end
  end

  `open #{elbowfile}`

  sscorefile = "examples/data/output/silhouette.png"

  Gnuplot.open do |gp|
    Gnuplot::Plot.new( gp ) do |plot|
      plot.terminal "png"
      plot.output sscorefile
      plot.xlabel "k"
      plot.ylabel "Silhouette score"
      
      plot.data << Gnuplot::DataSet.new([ks, silhouettes]) do |ds|
        ds.with = "lines"
        ds.notitle
      end
    end
  end

  `open #{sscorefile}`

  outfile = "examples/data/output/best_k.png"

  Gnuplot.open do |gp|
    Gnuplot::Plot.new(gp) do |plot|
      plot.terminal "png"
      plot.output outfile

      bestrun.clusters.each do |cluster|
        x = cluster.points.map { |p| p[0] }
        y = cluster.points.map { |p| p[1] }

        plot.data << Gnuplot::DataSet.new([x, y]) do |ds|
          ds.notitle
        end
      end
    end
  end

  `open #{outfile}`
end
