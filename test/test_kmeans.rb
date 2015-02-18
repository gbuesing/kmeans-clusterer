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

    assert_in_delta 8.0, kmeans.error
    assert_in_delta 0.873, kmeans.silhouette
  end

  def test_distance_calculation
    d = KMeansClusterer::Distance.euclidean NArray[1,1], NArray[2,2]
    assert_in_delta Math.sqrt(2), d
  end

  def test_distance_calculation_with_matrix
    d = KMeansClusterer::Distance.euclidean NArray[[1,1],[5,5]].to_f, NArray[2,2].to_f
    assert_in_delta Math.sqrt(2), d[0]
    assert_in_delta Math.sqrt(18), d[1]
  end

  def test_distance_calculation_with_matrix
    # [ [ 0.0, 1.41421, 2.82843, 5.65685, 12.7279 ], 
    # [ 5.65685, 4.24264, 2.82843, 0.0, 7.07107 ] ]
    d = KMeansClusterer::Distance.euclidean NMatrix[[1,1],[5,5]].to_f, NMatrix[[1,1],[2,2],[3,3],[5,5],[10,10]].to_f
    assert_in_delta 0.0, d[0,true][0]
    assert_in_delta Math.sqrt(32), d[0,true][1]
    assert_in_delta Math.sqrt(2), d[1,true][0]
    assert_in_delta Math.sqrt(18), d[1,true][1]
  end

  def test_scale_data
    input = [
      [1,5,10],
      [10,5,205]
    ]

    actual, mean, std = KMeansClusterer::Scaler.scale NMatrix.cast(input, NArray::DFLOAT)

    expected = [
      [ -1.0, 0.0, -1.0 ],
      [ 1.0, 0.0, 1.0 ]
    ]

    assert_equal [3,2], actual.shape
    assert_equal [expected[0]], actual[true, 0].to_a
    assert_equal [expected[1]], actual[true, 1].to_a

    assert_equal [5.5, 5.0, 107.5], mean.to_a
    assert_equal [4.5, 1.0, 97.5], std.to_a
  end


  def test_prediction_instance_init_with_custom_centroids
    km = KMeansClusterer.new init: [[2,2], [-2,-2]]
    predicted = km.predict [[3,3], [-3,-3]]
    assert_equal 0, predicted[0]
    assert_equal 1, predicted[1]
  end

end
