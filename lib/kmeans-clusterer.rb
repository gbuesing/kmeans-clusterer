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

    def recenter
      return 0 if @points.empty?
      old_centroid = @centroid
      @centroid = calculate_centroid_from_points
      Distance.call @centroid.data, old_centroid.data
    end

    def << point
      point.cluster = self
      @points << point
    end

    def reset_points
      @points = []
    end

    def sorted_points
      distances = points_distances_from(centroid)
      @points.sort_by.with_index {|c, i| distances[i] }
    end

    def sum_of_squares_error
      return 0 if @points.empty?
      distances = points_distances_from(centroid)
      (distances**2).sum
    end

    def sum_of_distances
      return 0 if @points.empty?
      points_distances_from(centroid).sum
    end

    def dissimilarity point
      distances = points_distances_from(point)
      distances.sum / distances.length.to_f
    end

    private
      def calculate_centroid_from_points
        data = CalculateCentroid.call points_narray
        Point.new data
      end

      def points_distances_from point
        Distance.call points_narray, point.data
      end

      def points_narray
        NArray.to_na @points.map(&:data)
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
    labels = opts[:labels] || []

    @points = data.map.with_index do |instance, i|
      Point.new instance, labels[i]
    end

    @points_matrix = opts[:points_matrix]
    @points_norms = opts[:points_norms]

    init_clusters
  end

  def run 
    start_time = Time.now
    @iterations, @runtime = 0, 0

    loop do
      @iterations +=1

      assign_points_to_clusters

      moves = clusters.map(&:recenter)

      break if moves.max < 0.001 # i.e., no movement
      break if @iterations >= 300

      clusters.each(&:reset_points)
    end

    @runtime =  Time.now - start_time
    self
  end

  def assign_points_to_clusters
    centroids = NMatrix.cast @clusters.map {|c| c.centroid.data}

    distances = Distance.call(centroids, @points_matrix, @points_norms)

    @points.each_with_index do |point, i|
      min_distance_index = distances[i, true].sort_index[0]
      cluster = @clusters[min_distance_index]
      cluster << point
    end
  end

  def error
    @clusters.map(&:sum_of_squares_error).reduce(:+)
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
      a = acluster.dissimilarity(point)
      b = bcluster.dissimilarity(point)
      (b - a) / [a,b].max
    end

    scores.reduce(:+) / scores.length # mean score for all points
  end

  private
    def init_clusters
      case @init
      when :random
        random_cluster_init
      when Array
        custom_cluster_init
      else
        kmpp_cluster_init
      end
    end

    # k-means++
    def kmpp_cluster_init
      @clusters = []
      pick = rand(@points.length)
      centroid = Point.new @points[pick].data.to_a
      @clusters << Cluster.new(centroid, 1)

      while @clusters.length < @k
        centroids = get_cluster_centroids

        d2 = @points.map do |point|
          dists = Distance.call centroids, point.data
          dists.min**2 # closest cluster distance, squared
        end

        d2 = NArray.to_na d2
        probs = d2 / d2.sum
        cumprobs = probs.cumsum
        r = rand
        # pick = cumprobs.to_a.index {|prob| r < prob }
        pick = (cumprobs >= r).where[0]
        centroid = Point.new @points[pick].data.to_a
        cluster = Cluster.new(centroid, @clusters.length + 1)
        @clusters << cluster
      end
    end

    def custom_cluster_init
      @clusters = @init.map.with_index do |instance, i|
        point = Point.new NArray.to_na(instance).to_f
        Cluster.new point, i+1
      end
    end

    def random_cluster_init
      @clusters = pick_k_random_points.map.with_index {|centroid, i| Cluster.new centroid, i+1 }
    end

    def pick_k_random_points
      pick_k_random_indexes.map {|i| Point.new @points[i].data.to_a }
    end

    def pick_k_random_indexes
      @points.length.times.to_a.shuffle.slice(0, @k)
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
