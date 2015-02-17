require 'narray'

class KMeansClusterer
  TYPECODE = { double: NArray::DFLOAT, single: NArray::SFLOAT }

  module Scaler
    def self.mean data
      data.mean(1)
    end

    def self.std data
      std = data.rmsdev(1)
      std[std.eq(0)] = 1.0 # so we don't divide by 0
      std
    end

    def self.scale data, mean = nil, std = nil, typecode = nil
      data = NArray.ref(data)
      mean ||= self.mean(data)
      std ||= self.std(data)
      data = (data - mean) / std
      [NMatrix.ref(data), mean, std]
    end
  end

  module Distance
    def self.euclidean x, y, yy = nil
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

    def sorted_points point = @centroid
      point = point.data if point.is_a?(Point)
      point = NArray.cast(point, @centroid.typecode) unless point.is_a?(NArray)
      points_data = NArray.cast(@points.map(&:data))
      distances = Distance.euclidean(points_data, point)
      @points.sort_by.with_index {|p, i| distances[i] }
    end
  end


  DEFAULT_OPTS = { scale_data: false, runs: 10, log: false, init: :kmpp, float_precision: :double }

  def self.run k, data, opts = {}
    opts = DEFAULT_OPTS.merge(opts)

    opts[:k] = k
    opts[:typecode] = TYPECODE[opts[:float_precision]]

    unless data.is_a?(NMatrix)
      data = NMatrix.cast data, opts[:typecode]
    end

    if opts[:scale_data]
      data, mean, std = Scaler.scale(data, nil, nil, opts[:typecode])
      opts[:mean] = mean
      opts[:std] = std
    end

    opts[:points_matrix] = data
    opts[:row_norms] = opts[:points_matrix].map {|v| v**2}.sum(0)

    bestrun = nil

    opts[:runs].times do |i|
      km = new(opts).run

      if opts[:log]
        puts "[#{i + 1}] #{km.iterations} iter\t#{km.runtime.round(2)}s\t#{km.error.round(2)} err"
      end
      
      if bestrun.nil? || (km.error < bestrun.error)
        bestrun = km
      end
    end

    bestrun.finish
  end


  attr_reader :k, :points, :clusters, :centroids, :error, :mean, :std, :iterations, :runtime


  def initialize opts = {}
    @k = opts[:k]
    @init = opts[:init]
    @labels = opts[:labels] || []
    @row_norms = opts[:row_norms]

    @points_matrix = opts[:points_matrix]
    @points_count = @points_matrix.shape[1] if @points_matrix
    @mean = opts[:mean]
    @std = opts[:std]
    @scale_data = opts[:scale_data]
    @typecode = opts[:typecode]

    init_centroids
  end

  def run 
    start_time = Time.now
    @iterations, @runtime = 0, 0

    @cluster_point_ids = Array.new(@k) { [] }

    loop do
      @iterations +=1

      distances = Distance.euclidean(@centroids, @points_matrix, @row_norms)

      # assign point ids to @cluster_point_ids
      @points_count.times do |i|
        min_distance_index = distances[i, true].sort_index[0]
        @cluster_point_ids[min_distance_index] << i
      end

      moves = []
      updated_centroids = []

      @k.times do |i|
        centroid = NArray.ref(@centroids[true, i].flatten)
        point_ids = @cluster_point_ids[i]

        if point_ids.empty?
          newcenter = centroid
          moves << 0
        else
          points = @points_matrix[true, point_ids]
          newcenter = points.mean(1)
          moves << Distance.euclidean(centroid, newcenter)
        end

        updated_centroids << newcenter
      end

      @centroids = NMatrix.cast updated_centroids, @typecode

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
    data = NMatrix.cast(data, @typecode)
    data, _m, _s = Scaler.scale(data, @mean, @std, @typecode) if @scale_data
    distances = Distance.euclidean(@centroids, data)
    data.shape[1].times.map do |i|
      distances[i, true].sort_index[0] # index of closest cluster
    end
  end

  def sorted_clusters point = origin
    data = point.is_a?(Point) ? point.data : NArray.cast(point, @typecode)
    distances = Distance.euclidean(NArray.ref(@centroids), data)
    @clusters.sort_by.with_index {|c, i| distances[i] }
  end

  def silhouette
    return 1.0 if @k < 2

    distances = Distance.euclidean(@centroids, @points_matrix, @row_norms)

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

  def inspect
    %{#<#{self.class.name} k:#{@k} iterations:#{@iterations} error:#{@error} runtime:#{@runtime}>}
  end

  private
    def wrap_point point
      return point if point.is_a?(Point)
      Point.new(0, NArray.cast(point, @typecode))
    end

    def dissimilarity points, point
      distances = Distance.euclidean points, point
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

        distances = Distance.euclidean(centroids, @points_matrix, @row_norms)

        d2 = []
        @points_count.times do |i|
          min_distance = distances[i, true].min
          d2 << min_distance**2
        end

        d2 = NArray.cast(d2, @typecode)
        probs = d2 / d2.sum
        cumprobs = probs.cumsum
        r = rand
        pick = (cumprobs >= r).where[0]
        centroid_ids << pick
      end

      @centroids = @points_matrix[true, centroid_ids]
    end

    def custom_centroid_init
      @centroids = NMatrix.cast @init, @typecode
      @k = @init.length
    end

    def random_centroid_init
      @centroids = @points_matrix[true, pick_k_random_indexes]
    end

    def pick_k_random_indexes
      @points_count.times.to_a.sample @k
    end

    def set_points
      @points = @points_count.times.map do |i|
        data = NArray.ref @points_matrix[true, i].flatten
        Point.new(i, data, @labels[i])
      end
    end

    def set_clusters
      @clusters = @k.times.map do |i|
        centroid = NArray.ref @centroids[true, i].flatten
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
          distances = Distance.euclidean points, centroid
          (distances**2).sum
        end
      end

      errors.reduce(:+)
    end

    def get_point i
      NArray.ref @points_matrix[true, i].flatten
    end

    def get_centroid i
      NArray.ref(@centroids[true, i].flatten)
    end

    def get_points_for_centroid i
      point_ids = @cluster_point_ids[i]
      points = @points_matrix[true, point_ids]
      points.empty? ? NArray.sfloat(0) : NArray.ref(points)
    end

    def origin
      wrap_point Array.new(@points[0].dimension, 0) 
    end
end
