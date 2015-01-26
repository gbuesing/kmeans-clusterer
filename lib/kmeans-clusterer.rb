require 'narray'

class KMeansClusterer

  # Euclidean distance function. Requires instances of NArray as args
  EuclideanDistance = -> (a, b) { NMath.sqrt ((a - b)**2).sum(0) }

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
    attr_reader :center, :points
    attr_accessor :label

    def initialize center, label = nil
      @center = center
      @label = label
      @points = []
    end

    def recenter
      if @points.empty?
        0
      else
        old_center = @center
        @center = calculate_center_from_points
        EuclideanDistance.call @center.data, old_center.data
      end
    end

    def << point
      point.cluster = self
      @points << point
    end

    def reset_points
      @points = []
    end

    def sorted_points
      distances = EuclideanDistance.call points_narray, center.data
      @points.sort_by.with_index {|c, i| distances[i] }
    end

    def sum_of_squares_error
      if @points.empty?
        0
      else
        distances = EuclideanDistance.call points_narray, center.data
        (distances**2).sum
      end
    end

    def dissimilarity point
      distances = EuclideanDistance.call points_narray, point.data
      distances.sum / distances.length.to_f
    end

    private
      def calculate_center_from_points
        mean = points_narray.mean(1)
        Point.new(mean)
      end

      def points_narray
        NArray.to_na @points.map(&:data)
      end
  end


  def self.run k, data, opts = {}
    data = data.map {|instance| NArray.to_na(instance) } # eagerly cast to NArray to reduce copies
    runcount = opts[:runs] || 8
    runs = runcount.times.map { new(k, data, opts).run }
    runs.sort_by(&:sum_of_squares_error).first
  end


  attr_reader :k, :points, :clusters, :iterations, :runtime


  def initialize k, data, opts = {}
    raise(ArgumentError, "k cannot be greater than the number of points") if k > data.length

    @k = k
    @init = opts[:init] || :kmpp
    labels = opts[:labels] || []

    @points = data.map.with_index do |instance, i|
      Point.new instance, labels[i]
    end

    @iterations, @runtime = 0, 0
  end

  def run 
    start_time = Time.now

    if @init == :kmpp
      kmpp_cluster_init
    else
      random_cluster_init
    end

    loop do
      @iterations +=1

      @points.each do |point|
        cluster = closest_cluster(point)
        cluster << point
      end

      moves = clusters.map(&:recenter)

      break if moves.max < 0.001 # i.e., no movement
      break if @iterations >= 300

      clusters.each(&:reset_points)
    end

    @runtime =  Time.now - start_time
    self
  end

  def sum_of_squares_error
    @clusters.map(&:sum_of_squares_error).reduce(:+)
  end

  def closest_cluster point = origin
    sorted_clusters(point).first
  end

  def sorted_clusters point = origin
    centers = NArray.to_na @clusters.map {|c| c.center.data }
    distances = EuclideanDistance.call(centers, point.data)
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
    # k-means++
    def kmpp_cluster_init
      @clusters = []
      pick = rand(@points.length)
      center = Point.new @points[pick].data.to_a
      @clusters << Cluster.new(center)

      while @clusters.length < @k
        centers = NArray.to_na @clusters.map {|c| c.center.data }

        d2 = @points.map do |point|
          dists = EuclideanDistance.call centers, point.data
          dists.min**2 # closest cluster distance, squared
        end

        d2 = NArray.to_na d2
        probs = d2 / d2.sum
        cumprobs = probs.cumsum
        r = rand
        pick = cumprobs.to_a.index {|prob| r < prob }
        center = Point.new @points[pick].data.to_a
        @clusters << Cluster.new(center)
      end
    end

    def random_cluster_init
      @clusters = pick_k_random_points.map.with_index {|center, i| Cluster.new center, i+1 }
    end

    def pick_k_random_points
      pick_k_random_indexes.map {|i| Point.new @points[i].data.to_a }
    end

    def pick_k_random_indexes
      @points.length.times.to_a.shuffle.slice(0, @k)
    end
end
