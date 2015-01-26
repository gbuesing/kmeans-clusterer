require 'rubygems'
require 'bundler/setup'
require 'minitest/autorun'
require_relative '../lib/kmeans-clusterer'


class TestKMeansClusterer < MiniTest::Test

  def test_clustering
    data = [
      [3, 3], [-3, 3], [3, -3], [-3, -3],
      [3, 4], [-3, 4], [3, -4], [-3, -4],
      [4, 3], [-4, 3], [4, -3], [-4, -3],
      [4, 4], [-4, 4], [4, -4], [-4, -4],
    ]

    kmeans = KMeansClusterer.run 4, data
    
    kmeans.clusters.each do |cluster|
      xs, ys = cluster.points.map {|p| p[0]}, cluster.points.map {|p| p[1]}
      assert (xs.inject(1) {|m, v| m * v}) > 0 # i.e., ensure xs are all same sign
      assert (ys.inject(1) {|m, v| m * v}) > 0 # i.e., ensure ys are all same sign
    end

    assert_in_delta 8.0, kmeans.sum_of_squares_error
    assert_in_delta 0.873, kmeans.silhouette_score
  end

end


class TestCluster < MiniTest::Test

  def test_recenter
    c = KMeansClusterer::Cluster.new KMeansClusterer::Point.new([-5,-7])
    p1 = KMeansClusterer::Point.new [1,2]
    p2 = KMeansClusterer::Point.new [6, 5]
    c << p1
    c << p2
    dist = c.recenter
    assert dist > 0
    x, y = c.center[0], c.center[1]
    assert_equal 3.5, x
    assert_equal 3.5, y
  end

  def test_sum_of_squares_error
    c = KMeansClusterer::Cluster.new KMeansClusterer::Point.new([-5,-7])
    p1 = KMeansClusterer::Point.new [1,2]
    p2 = KMeansClusterer::Point.new [6, 5]
    c << p1
    c << p2
    assert_equal 382.0, c.sum_of_squares_error
  end

  def test_sum_of_squares_error_when_no_points
    c = KMeansClusterer::Cluster.new KMeansClusterer::Point.new([-5,-7])
    assert_equal 0, c.sum_of_squares_error
  end

  def test_dissimilarity
    c1 = KMeansClusterer::Cluster.new KMeansClusterer::Point.new([3,3])
    p1 = KMeansClusterer::Point.new [1,2]
    p2 = KMeansClusterer::Point.new [6, 5]
    c1 << p1
    c1 << p2

    p3 = KMeansClusterer::Point.new [-7, -8]
    assert c1.dissimilarity(p3) > c1.dissimilarity(p2)
  end

end
