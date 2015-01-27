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

  def test_distance_calculation
    d = KMeansClusterer::Distance.call NArray[1,1], NArray[2,2]
    assert_in_delta Math.sqrt(2), d
  end

  def test_distance_calculation_with_matrix
    d = KMeansClusterer::Distance.call NArray[[1,1],[5,5]], NArray[2,2]
    assert_in_delta Math.sqrt(2), d[0]
    assert_in_delta Math.sqrt(18), d[1]
  end

  def test_centroid_calculation
    c = KMeansClusterer::Centroid.call NArray[[1,1],[5,5],[4,6]]
    assert_in_delta (1+5+4)/3.0, c[0]
    assert_in_delta (1+5+6)/3.0, c[1]
  end

  def test_scale_data
    input = [
      [1,5,10],
      [10,5,205]
    ]

    actual = KMeansClusterer.scale_data input

    expected = [
      [ -1.0, 0.0, -1.0 ],
      [ 1.0, 0.0, 1.0 ]
    ]

    assert_equal expected.length, actual.length
    assert_equal expected[0], actual[0].to_a
    assert_equal expected[1], actual[1].to_a
  end

end

class TestKMediansClusterer < MiniTest::Test

  def test_clustering
    data = [
      [3, 3], [-3, 3], [3, -3], [-3, -3],
      [3, 4], [-3, 4], [3, -4], [-3, -4],
      [4, 3], [-4, 3], [4, -3], [-4, -3],
      [4, 4], [-4, 4], [4, -4], [-4, -4],
    ]

    kmedians = KMediansClusterer.run 4, data
    
    kmedians.clusters.each do |cluster|
      xs, ys = cluster.points.map {|p| p[0]}, cluster.points.map {|p| p[1]}
      assert (xs.inject(1) {|m, v| m * v}) > 0 # i.e., ensure xs are all same sign
      assert (ys.inject(1) {|m, v| m * v}) > 0 # i.e., ensure ys are all same sign
    end
  end

  def test_distance_calculation
    d = KMediansClusterer::Distance.call NArray[1,1].to_f, NArray[2,2].to_f
    assert_equal 2, d
  end

  def test_distance_calculation_with_matrix
    d = KMediansClusterer::Distance.call NArray[[1,1],[5,5]].to_f, NArray[2,2].to_f
    assert_equal 2, d[0]
    assert_equal 6, d[1]
  end

  def test_centroid_calculation_with_odd_count
    c = KMediansClusterer::Centroid.call NArray[[1,1],[5,5],[4,6]].to_f
    assert_equal 4, c[0]
    assert_equal 5, c[1]
  end

  def test_centroid_calculation_with_even_count
    c = KMediansClusterer::Centroid.call NArray[[1,1],[5,5],[4,6],[7,-2]].to_f
    assert_equal 4.5, c[0]
    assert_equal 3.0, c[1]
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
    x, y = c.centroid[0], c.centroid[1]
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
