require 'narray'

class KMeansClusterer

  CalculateCentroid = -> (a) { a.mean(1) }

  Distance = -> (x, y, yy = nil) do 
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

  class Point
    attr_reader :data
    attr_accessor :cluster, :label

    def initialize data, label = nil
      @data = NArray.to_na data
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
    attr_reader :centroid, :points
    attr_accessor :label

    def initialize centroid, label = nil
      @centroid = centroid
      @label = label
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


  def self.run k, data, opts = {}
    raise(ArgumentError, "k cannot be greater than the number of points") if k > data.length

    data = if opts[:scale_data]
      scale_data data
    else
      data.map {|row| NArray.to_na(row).to_f}
    end

    runcount = opts[:runs] || 10
    errors = []

    opts[:points_matrix] = NMatrix.cast data
    opts[:points_norms] = opts[:points_matrix].map {|v| v**2}.sum(0)


    runs = runcount.times.map do |i|
      km = new(k, data, opts).run
      error = km.error
      if opts[:log]
        puts "[#{i + 1}] #{km.iterations} iter\t#{km.runtime.round(2)}s\t#{error.round(2)} err"
      end
      errors << error
      km
    end

    runs.sort_by.with_index {|run, i| errors[i] }.first
  end

  # see scikit-learn scale and _mean_and_std methods
  def self.scale_data data
    nadata = NArray.to_na(data).to_f
    mean = nadata.mean(1)
    std = nadata.rmsdev(1)
    std[std.eq(0)] = 1.0 # so we don't divide by 0
    nadata = (nadata - mean) / std
    # convert back to an array, containing NArrays for each row
    data.length.times.map {|i| nadata[true, i] }
  end


  attr_reader :k, :points, :clusters, :iterations, :runtime


  def initialize k, data, opts = {}
    @k = k
    @init = opts[:init] || :kmpp
    @labels = opts[:labels] || []

    # @points = data.map.with_index do |instance, i|
    #   Point.new instance, labels[i]
    # end

    @points_matrix = opts[:points_matrix]
    @points_norms = opts[:points_norms]
    @points_count = @points_matrix.shape[1]

    init_centroids
  end

  def run 
    start_time = Time.now
    @iterations, @runtime = 0, 0

    @cluster_point_ids = Array.new(@k) { [] }

    loop do
      @iterations +=1

      distances = Distance.call(@centroids, @points_matrix, @points_norms)

      # assign point ids to @cluster_point_ids
      @points_count.times do |i|
        min_distance_index = distances[i, true].sort_index[0]
        @cluster_point_ids[min_distance_index] << i
      end

      # moves = clusters.map(&:recenter)
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
          moves << Distance.call(centroid, newcenter)
        end
        updated_centroids << newcenter
      end

      @centroids = NMatrix.cast updated_centroids

      break if moves.max < 0.001 # i.e., no movement
      break if @iterations >= 300

      # clusters.each(&:reset_points)
      @cluster_point_ids = Array.new(@k) { [] }
    end

    @points = @points_count.times.map do |i|
      data = NArray.cast @points_matrix[true, i].flatten
      Point.new(data, @labels[i])
    end

    @clusters = @k.times.map do |i|
      centroid = NArray.cast @centroids[true, i].flatten
      c = Cluster.new Point.new(centroid), i + 1
      @cluster_point_ids[i].each do |p|
        c << @points[p]
      end
      c
    end

    @runtime =  Time.now - start_time
    self
  end

  def error
    errors = @clusters.map do |c|
      if c.points.empty?
        0
      else
        distances = Distance.call NArray.cast(c.points.map(&:data)), c.centroid.data
        (distances**2).sum
      end
    end

    errors.reduce(:+)
  end

  def closest_cluster point = origin
    sorted_clusters(point).first
  end

  def sorted_clusters point = origin
    point = Point.new(point) unless point.is_a?(Point)
    centroids = get_cluster_centroids
    distances = Distance.call(centroids, point.data)
    @clusters.sort_by.with_index {|c, i| distances[i] }
  end

  def origin
    Point.new Array.new(@points[0].dimension, 0)
  end

  def silhouette_score
    return 1.0 if @clusters.length < 2
    
    scores = @points.map do |point|
      acluster, bcluster = sorted_clusters(point).slice(0,2)
      a = dissimilarity(acluster.points_narray, point.data)
      b = dissimilarity(bcluster.points_narray, point.data)
      (b - a) / [a,b].max
    end

    scores.reduce(:+) / scores.length # mean score for all points
  end

  private
    def dissimilarity points, point
      distances = Distance.call points, point
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

        distances = Distance.call(centroids, @points_matrix, @points_norms)

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
end

# class KMediansClusterer < KMeansClusterer
#   Distance = -> (x, y, yy = nil) { (x - y).abs.sum(0) }
#   CalculateCentroid = -> (a) { a.rot90.median(0) }

#   def error
#     @clusters.map(&:sum_of_distances).reduce(:+)
#   end
# end
