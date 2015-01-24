require 'narray'

class KMeansClusterer

  class Point
    attr_reader :data
    attr_accessor :cluster, :tag

    def initialize data
      @data = NArray.to_na data
    end

    def [] index
      @data[index]
    end

    def dimension
      @data.length
    end

    def distance_from point
      euclidean_distance @data, point.data
    end

    private
      def euclidean_distance vec1, vec2
        Math.sqrt ((vec1 - vec2)**2).sum
      end
  end


  class Cluster
    attr_reader :center, :points
    attr_accessor :tag

    def initialize center
      @center = center
      @points = []
    end

    def recenter
      if @points.empty?
        0
      else
        old_center = @center
        @center = calculate_center_from_points
        distance_from_center old_center
      end
    end

    def add point
      point.cluster = self
      @points << point
    end

    def reset_points
      @points = []
    end

    def distance_from_center point
      @center.distance_from point
    end

    def sorted_points
      points.sort_by {|point| distance_from_center(point) }
    end

    def sum_of_squares_error
      errors = points.map {|point| distance_from_center(point) }
      (NArray.to_na(errors)**2).sum
    end

    private
      def calculate_center_from_points
        mean = NArray[@points.map(&:data)].mean(1)
        Point.new(mean)
      end
  end


  def self.run k, points, opts = {}
    points = points.map {|data| NArray.to_na(data) } # eagerly cast to NArray to reduce copies
    runs = opts[:runs] || 8
    outputs = runs.times.map { new(k, points, opts).run }
    outputs.sort_by {|output| output.sum_of_squares_error }.first
  end


  attr_reader :k, :points, :clusters, :iterations, :runtime


  def initialize k, points, opts = {}
    raise(ArgumentError, "k cannot be greater than the number of points") if k > points.length

    @k = k
    @random = Random.new(opts[:random_seed] || Random.new_seed)
    tags = opts[:tags] || []

    @points = points.map.with_index do |data, i|
      Point.new(data).tap {|p| p.tag = tags[i] }
    end

    @iterations, @runtime = 0, 0
  end

  def run 
    start_time = Time.now

    @clusters = pick_k_random_points.map {|point| Cluster.new(point) }

    loop do
      @iterations +=1

      @points.each do |point|
        cluster = closest_cluster(point)
        cluster.add point
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
    @clusters.sort_by {|c| c.distance_from_center(point) }
  end

  def origin
    Point.new Array.new(@points[0].dimension, 0)
  end

  private
    def pick_k_random_points
      indexes = pick_k_random_indexes
      datas = indexes.map {|i| @points[i].data.to_a }
      datas.map {|data| Point.new(data) }
    end

    def pick_k_random_indexes
      indexes = @points.length.times.to_a
      indexes.shuffle! random: @random
      indexes.slice(0, @k)
    end
end
