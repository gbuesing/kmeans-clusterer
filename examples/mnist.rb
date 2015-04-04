#!/usr/bin/env ruby

require 'rubygems'
require 'bundler/setup'
require_relative '../lib/kmeans-clusterer'
require_relative './utils/mnist_loader'
require 'narray'
require 'optparse'
require 'pca'

k = 10
train_size = 5000
test_size = 200
runs = 1 # not much seems to be gained by multiple runs for this example
skip_plot = false
pca_components = nil

OptionParser.new do |opts|
  opts.on("-kK") {|v| k = v.to_i }
  opts.on("-nN") {|v| train_size = v.to_i }
  opts.on("--skip-plot") {|v| skip_plot = true }
  opts.on("--pca-components=N") {|v| pca_components = v.to_i }
end.parse!

orig_data, labels = MnistLoader.training_set.get_data_and_labels(train_size + test_size)

# crop 4px border
data = orig_data.map do |row|
  row = NArray.to_na(row)
  row = row.reshape!(28,28)
  row = row[4..23, 4..23]
  row.reshape!(20*20).to_a
end



train_data, train_labels = data.slice(0, train_size), labels.slice(0, train_size)
test_data, test_labels = data.slice(train_size, test_size), labels.slice(train_size, test_size)

if pca_components
  puts "\nRunning PCA with components=#{pca_components}..."
  t = Time.now
  pca = PCA.new components: pca_components
  train_data = pca.fit_transform train_data
  elapsed = Time.now - t
  puts "\nPCA finished. Time #{elapsed.round(2)}s"
end

puts "\nClustering #{train_size} images:"

t = Time.now
kmeans = KMeansClusterer.run(k, train_data, labels: train_labels, runs: runs, log: true)
elapsed = Time.now - t

# kmeans.clusters.each do |cluster|
#   puts "\n#---\n\n"
#   puts cluster.sorted_points.map(&:label).join(' ')
# end

puts "\nBest of #{runs} runs (total time #{elapsed.round(2)}s):"
puts "10 clusters in #{kmeans.iterations} iterations, #{kmeans.runtime.round(2)}s"

puts "\nUsing kmeans to cluster #{test_size} samples from test set:\n\n"

# console output: show lables

predictions_labels = Array.new(k) { [] }

if pca_components
  test_data = pca.transform test_data
end

predictions = kmeans.predict test_data

predictions.each_with_index do |predict, i|
  label = test_labels[i]
  predictions_labels[predict] << label
end

predictions_labels.each do |vals|
  # puts "\n#---\n\n"
  puts vals.join(' ')
end


# png output: show actual images

def image_grid data_groups, filename
  image_size = 28
  max_per_row = 25
  gridrows, gridcols = data_groups.length, 25

  @png = ChunkyPNG::Image.new(gridcols * image_size, gridrows * image_size, ChunkyPNG::Color::TRANSPARENT)

  data_groups.each.with_index do |images, gridrow|
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

  # image_path = "examples/data/output/mnist_#{train_size}_#{runs}.png"
  puts "\nSaving png to #{filename}"
  @png.save filename, :compression => Zlib::NO_COMPRESSION

  `open #{filename}`
end

unless skip_plot
  require 'chunky_png'

  predictions_images = Array.new(k) { [] }

  orig_test = orig_data.slice(train_size, test_size)

  predictions.each_with_index do |predict, i|
    label = test_labels[i]
    predictions_images[predict] << orig_test[i]
  end

  image_grid predictions_images, "examples/data/output/mnist_#{train_size}_#{runs}.png"

  orig_predictions = kmeans.clusters.map do |cluster| 
    cluster.points.slice(0,25).map {|p| orig_data[p.id] }
  end

  image_grid orig_predictions, "examples/data/output/mnist_#{train_size}_#{runs}_closest.png"
end

puts "\nSilhouette score: #{kmeans.silhouette.round(2)}"

