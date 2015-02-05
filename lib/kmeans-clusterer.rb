require 'narray'

class KMeansClusterer
  module Scaler
    def self.mean data
      data.mean(1)
    end

    def self.std data
      std = data.rmsdev(1)
      std[std.eq(0)] = 1.0 # so we don't divide by 0
      std
    end

    def self.scale data, mean = nil, std = nil
      data = NArray.cast(data, NArray::DFLOAT)
      mean ||= self.mean(data)
      std ||= self.std(data)
      data = (data - mean) / std
      [data, mean, std]
    end
  end


  class Point
    attr_reader :id, :data
    attr_accessor :cluster, :label

    def initialize id, data, label = nil
      @id = id
      @data = data
      @label = label
    end

    def [] index
      @data[index]
    end

    def to_a
      @data.to_a
    end

    def to_s
      to_a.to_s
    end

    def dimension
      @data.length
    end
  end


  class Cluster
    attr_reader :id, :centroid, :points
    attr_accessor :label

    def initialize id, centroid
      @id = id
      @centroid = centroid
      @points = []
    end

    def << point
      point.cluster = self
      @points << point
    end

    def points_narray
      NArray.cast @points.map(&:data)
    end
  end


  DEFAULT_OPTS = { scale_data: false, runs: 10, log: false, init: :kmpp}

  def self.run k, data, opts = {}
    opts = DEFAULT_OPTS.merge(opts)

    if opts[:scale_data]
      data, mean, std = Scaler.scale(data)
      opts[:mean] = mean
      opts[:std] = std
    end

    points_matrix = NMatrix.cast(data, NArray::DFLOAT)
    opts[:row_norms] = points_matrix.map {|v| v**2}.sum(0)

    runs = opts[:runs].times.map do |i|
      km = new(k, points_matrix, opts).run
      if opts[:log]
        puts "[#{i + 1}] #{km.iterations} iter\t#{km.runtime.round(2)}s\t#{km.error.round(2)} err"
      end
      km
    end

    runs.sort_by {|run| run.error }.first.finish
  end


  attr_reader :k, :points, :clusters, :error, :iterations, :runtime


  def initialize k, points_matrix, opts = {}
    @k = k
    @init = opts[:init] || :kmpp
    @labels = opts[:labels] || []
    @row_norms = opts[:row_norms]

    @points_matrix = points_matrix
    @points_count = @points_matrix.shape[1]
    @mean = opts[:mean]
    @std = opts[:std]
    @scale_data = opts[:scale_data]

    init_centroids
  end

  def run 
    start_time = Time.now
    @iterations, @runtime = 0, 0

    @cluster_point_ids = Array.new(@k) { [] }

    loop do
      @iterations +=1

      distances = distance(@centroids, @points_matrix)

      # assign point ids to @cluster_point_ids
      @points_count.times do |i|
        min_distance_index = distances[i, true].sort_index[0]
        @cluster_point_ids[min_distance_index] << i
      end

      moves = []
      updated_centroids = []

      @k.times do |i|
        centroid = NArray.cast(@centroids[true, i].flatten)
        point_ids = @cluster_point_ids[i]

        if point_ids.empty?
          newcenter = centroid
          moves << 0
        else
          points = @points_matrix[true, point_ids]
          newcenter = points.mean(1)
          moves << distance(centroid, newcenter)
        end

        updated_centroids << newcenter
      end

      @centroids = NMatrix.cast updated_centroids

      break if moves.max < 0.001 # i.e., no movement
      break if @iterations >= 300

      @cluster_point_ids = Array.new(@k) { [] }
    end

    @error = calculate_error
    @runtime =  Time.now - start_time
    self
  end

  def finish
    set_points
    set_clusters
    self
  end

  def predict data
    data, _m, _s = Scaler.scale(data, @mean, @std) if @scale_data
    data = NMatrix.cast(data, NArray::DFLOAT)
    distances = distance(@centroids, data, nil)
    data.shape[1].times.map do |i|
      distances[i, true].sort_index[0] # index of closest cluster
    end
  end

  def sorted_clusters point = origin
    point = wrap_point point
    centroids = get_cluster_centroids
    distances = distance(centroids, point.data)
    @clusters.sort_by.with_index {|c, i| distances[i] }
  end

  def origin
    wrap_point Array.new(@points[0].dimension, 0) 
  end

  def silhouette_score
    return 1.0 if @k < 2

    distances = distance(@centroids, @points_matrix)

    scores = @points_count.times.map do |i|
      point = get_point i
      cluster_indexes = distances[i, true].sort_index

      c1_points = get_points_for_centroid cluster_indexes[0]
      c2_points = get_points_for_centroid cluster_indexes[1]

      a = dissimilarity(c1_points, point)
      b = dissimilarity(c2_points, point)
      (b - a) / [a,b].max
    end

    scores.reduce(:+) / scores.length # mean score for all points
  end

  private
    def wrap_point point
      return point if point.is_a?(Point)
      Point.new(0, NArray.to_na(point).to_f)
    end

    def dissimilarity points, point
      distances = distance points, point
      distances.sum / distances.length.to_f
    end

    def init_centroids
      case @init
      when :random
        random_centroid_init
      when Array
        custom_centroid_init
      else
        kmpp_centroid_init
      end
    end

    # k-means++
    def kmpp_centroid_init
      centroid_ids = []
      pick = rand(@points_count)
      centroid_ids << pick

      while centroid_ids.length < @k
        centroids = @points_matrix[true, centroid_ids]

        distances = distance(centroids, @points_matrix)

        d2 = []
        @points_count.times do |i|
          min_distance = distances[i, true].min
          d2 << min_distance**2
        end

        d2 = NArray.to_na d2
        probs = d2 / d2.sum
        cumprobs = probs.cumsum
        r = rand
        pick = (cumprobs >= r).where[0]
        centroid_ids << pick
      end

      @centroids = @points_matrix[true, centroid_ids]
    end

    def custom_centroid_init
      @centroids = NMatrix.cast @init
    end

    def random_centroid_init
      @centroids = @points_matrix[true, pick_k_random_indexes]
    end

    def pick_k_random_indexes
      @points_count.times.to_a.shuffle.slice(0, @k)
    end

    def get_cluster_centroids
      NArray.to_na @clusters.map {|c| c.centroid.data }
    end

    def set_points
      @points = @points_count.times.map do |i|
        data = NArray.cast @points_matrix[true, i].flatten
        Point.new(i, data, @labels[i])
      end
    end

    def set_clusters
      @clusters = @k.times.map do |i|
        centroid = NArray.cast @centroids[true, i].flatten
        c = Cluster.new i, Point.new(-i, centroid)
        @cluster_point_ids[i].each do |p|
          c << @points[p]
        end
        c
      end
    end

    def calculate_error
      errors = @k.times.map do |i|
        centroid = get_centroid i
        points = get_points_for_centroid i

        if points.empty?
          0
        else
          distances = distance points, centroid
          (distances**2).sum
        end
      end

      errors.reduce(:+)
    end

    def get_point i
      NArray.cast @points_matrix[true, i].flatten
    end

    def get_centroid i
      NArray.cast(@centroids[true, i].flatten)
    end

    def get_points_for_centroid i
      point_ids = @cluster_point_ids[i]
      NArray.cast @points_matrix[true, point_ids]
    end

    def distance x, y, yy = @row_norms
      if x.is_a?(NMatrix) && y.is_a?(NMatrix)
        xx = x.map {|v| v**2}.sum(0)
        yy ||= y.map {|v| v**2}.sum(0)
        xy = x * y.transpose
        distance = xy * -2
        distance += xx
        distance += yy.transpose
        NMath.sqrt distance
      else
        NMath.sqrt ((x - y)**2).sum(0)
      end
    end
end

# class KMediansClusterer < KMeansClusterer
#   Distance = -> (x, y, yy = nil) { (x - y).abs.sum(0) }
#   CalculateCentroid = -> (a) { a.rot90.median(0) }

#   def error
#     @clusters.map(&:sum_of_distances).reduce(:+)
#   end
# end
