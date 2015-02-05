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
    assert_in_delta 0.873, kmeans.silhouette_score
  end

  def test_distance_calculation
    km = KMeansClusterer.new(1, NArray[], points_matrix: NMatrix[[1]]) 
    d = km.send :distance, NArray[1,1], NArray[2,2]
    assert_in_delta Math.sqrt(2), d
  end

  def test_distance_calculation_with_matrix
    km = KMeansClusterer.new(1, NArray[], points_matrix: NMatrix[[1]]) 
    d = km.send :distance, NArray[[1,1],[5,5]].to_f, NArray[2,2].to_f
    assert_in_delta Math.sqrt(2), d[0]
    assert_in_delta Math.sqrt(18), d[1]
  end

  def test_distance_calculation_with_matrix
    # [ [ 0.0, 1.41421, 2.82843, 5.65685, 12.7279 ], 
    # [ 5.65685, 4.24264, 2.82843, 0.0, 7.07107 ] ]
    km = KMeansClusterer.new(1, NArray[], points_matrix: NMatrix[[1]]) 
    d = km.send :distance, NMatrix[[1,1],[5,5]].to_f, NMatrix[[1,1],[2,2],[3,3],[5,5],[10,10]].to_f
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

# class TestKMediansClusterer < MiniTest::Test

#   def test_clustering
#     data = [
#       [3, 3], [-3, 3], [3, -3], [-3, -3],
#       [3, 4], [-3, 4], [3, -4], [-3, -4],
#       [4, 3], [-4, 3], [4, -3], [-4, -3],
#       [4, 4], [-4, 4], [4, -4], [-4, -4],
#     ]

#     kmedians = KMediansClusterer.run 4, data
    
#     kmedians.clusters.each do |cluster|
#       xs, ys = cluster.points.map {|p| p[0]}, cluster.points.map {|p| p[1]}
#       assert (xs.inject(1) {|m, v| m * v}) > 0 # i.e., ensure xs are all same sign
#       assert (ys.inject(1) {|m, v| m * v}) > 0 # i.e., ensure ys are all same sign
#     end

#     assert_in_delta 11.314, kmedians.error
#     assert_in_delta 0.873, kmedians.silhouette_score
#   end

#   def test_distance_calculation
#     d = KMediansClusterer::Distance.call NArray[1,1].to_f, NArray[2,2].to_f
#     assert_equal 2, d
#   end

#   def test_distance_calculation_with_matrix
#     d = KMediansClusterer::Distance.call NArray[[1,1],[5,5]].to_f, NArray[2,2].to_f
#     assert_equal 2, d[0]
#     assert_equal 6, d[1]
#   end

#   def test_centroid_calculation_with_odd_count
#     c = KMediansClusterer::CalculateCentroid.call NArray[[1,1],[5,5],[4,6]].to_f
#     assert_equal 4, c[0]
#     assert_equal 5, c[1]
#   end

#   def test_centroid_calculation_with_even_count
#     c = KMediansClusterer::CalculateCentroid.call NArray[[1,1],[5,5],[4,6],[7,-2]].to_f
#     assert_equal 4.5, c[0]
#     assert_equal 3.0, c[1]
#   end

# end


