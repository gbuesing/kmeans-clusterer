require 'rubygems'
require 'bundler/setup'
require 'minitest/autorun'
require_relative '../lib/kmeans-clusterer'


class TestKMeansClusterer < MiniTest::Test

  def test_clustering
    data = [
      [7,8],
      [-1,-4],
      [3,2],
      [-6,-9]
    ] 
    kmeans = KMeansClusterer.new(2, data, random_seed: 42)
    kmeans.run
    assert_equal kmeans.points[0].cluster, kmeans.points[2].cluster
    assert_equal kmeans.points[1].cluster, kmeans.points[3].cluster
    assert_equal 51.0, kmeans.sum_of_squares_error
  end

end


class TestPoint < MiniTest::Test

  def test_distance_from
    p1 = KMeansClusterer::Point.new [1,2]
    p2 = KMeansClusterer::Point.new [6, 5]
    assert_in_delta 5.83, p1.distance_from(p2)
  end

end


class TestCluster < MiniTest::Test

  def test_recenter
    c = KMeansClusterer::Cluster.new KMeansClusterer::Point.new([-5,-7])
    p1 = KMeansClusterer::Point.new [1,2]
    p2 = KMeansClusterer::Point.new [6, 5]
    c.add p1
    c.add p2
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
    c.add p1
    c.add p2
    assert_equal 382.0, c.sum_of_squares_error
  end

end
