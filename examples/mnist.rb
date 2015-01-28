#!/usr/bin/env ruby

require 'rubygems'
require 'bundler/setup'
require_relative '../lib/kmeans-clusterer'
require_relative './utils/mnist_loader'
require 'narray'
require 'chunky_png'



k = (ARGV[0] || 10).to_i
train_size = 5000
test_size = (ARGV[1] || 200).to_i

runs = 1 # not much seems to be gained by multiple runs for this example



orig_data, labels = MnistLoader.training_set.get_data_and_labels(train_size + test_size)

# crop 4px border
data = orig_data.map do |row|
  row = NArray.to_na(row)
  row = row.reshape!(28,28)
  row = row[4..23, 4..23]
  row.reshape!(20*20)
end



train_data, train_labels = data.slice(0, train_size), labels.slice(0, train_size)
test_data, test_labels = data.slice(train_size, test_size), labels.slice(train_size, test_size)

puts "Clustering #{train_size} images:"

t = Time.now
kmeans = KMeansClusterer.run(k, train_data, labels: train_labels, runs: runs, log: true)
elapsed = Time.now - t

# kmeans.clusters.each do |cluster|
#   puts "\n#---\n\n"
#   puts cluster.sorted_points.map(&:label).join(' ')
# end

puts "\nBest of #{runs} runs (total time #{elapsed.round(2)}s):"
puts "10 clusters in #{kmeans.iterations} iterations, #{kmeans.runtime.round(2)}s"
# puts "Silhouette score: #{kmeans.silhouette_score.round(2)}"

puts "\nUsing kmeans to cluster #{test_size} samples from test set:\n\n"

# console output: show lables

predictions_labels = Array.new(k) { [] }

test_data.each.with_index do |row, i|
  label = test_labels[i]
  cluster = kmeans.closest_cluster row
  predictions_labels[cluster.label - 1] << label
end

predictions_labels.each do |vals|
  # puts "\n#---\n\n"
  puts vals.join(' ')
end


# png output: show actual images

predictions_images = Array.new(k) { [] }

orig_test = orig_data.slice(train_size, test_size)

test_data.each.with_index do |row, i|
  cluster = kmeans.closest_cluster row
  predictions_images[cluster.label - 1] << orig_test[i]
end

image_size = 28
max_per_row = 25
gridrows, gridcols = k, 25

@png = ChunkyPNG::Image.new(gridcols * image_size, gridrows * image_size, ChunkyPNG::Color::TRANSPARENT)

predictions_images.each.with_index do |images, gridrow|
  gridrow_offset = gridrow * image_size
  images.slice(0,max_per_row).each.with_index do |image, gridcol|
    gridcol_offset = gridcol * image_size

    img_row = 0

    (image_size * image_size).times do |p|
      if p > 0 && p % image_size == 0
        img_row += 1
      end

      img_col = p % image_size

      pixel = image[p]
      @png[img_col + gridcol_offset, img_row + gridrow_offset] = ChunkyPNG::Color("black @ #{pixel}")
    end
  end
end

image_path = "examples/data/output/mnist_#{train_size}_#{runs}.png"
puts "\nSaving png to #{image_path}"
@png.save image_path, :compression => Zlib::NO_COMPRESSION

`open #{image_path}`

