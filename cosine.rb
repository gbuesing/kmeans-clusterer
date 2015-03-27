require 'narray'
require 'minitest/autorun'

    # Returns
    # -------
    # kernel matrix : array
    #     An array with shape (n_samples_X, n_samples_Y).
    # """
    # # to avoid recursive import

    # X, Y = check_pairwise_arrays(X, Y)

    # X_normalized = normalize(X, copy=True)
    # if X is Y:
    #     Y_normalized = X_normalized
    # else:
    #     Y_normalized = normalize(Y, copy=True)

    # K = safe_sparse_dot(X_normalized, Y_normalized.T, dense_output=True)

    # return K

def row_norms data
  squared_data = NArray.ref(data)**2
  NMatrix.ref(squared_data).sum(0)
end

def normalize x
  norm = NMath.sqrt row_norms(x)
  norm[norm.eq(0)] = 1.0
  x.dup.div! norm
end

def cosine_distance a, b
  if a.is_a?(NMatrix) && b.is_a?(NMatrix)
    a_normalized = normalize a
    b_normalized = normalize b
    sim = a_normalized * b_normalized.transpose
    sim *= -1
    sim.add! 1
    sim
  else
    a_norm = NMath.sqrt (a**2).sum(0)
    b_norm = NMath.sqrt (b**2).sum(0)
    denom = a_norm * b_norm
    if denom.is_a?(Float)
      return 0.0 if denom == 0
    else
      denom[denom.eq(0.0)] = 1.0  
    end
    sim = (a * b).sum(0) / denom
    sim *= -1
    if sim.is_a?(NArray)
      sim[sim.ne(0.0)] += 1
    else
      sim += 1
    end
    sim
  end
end

# def cosine_distance a, b
#   sim = cosine_similarity(a,b)
#   sim *= -1
#   sim += 1
#   sim
# end


class TestCosine < MiniTest::Test

  def test_with_narray
    d = cosine_distance NArray[1,2,3], NArray[3,5,7]
    assert_in_delta 0.0026, d
  end

  def test_narray_with_zero
    d = cosine_distance NArray[0,0,0], NArray[3,5,7]
    assert_equal 0.0, d
  end

  def test_multi_narray_with_zero
    d = cosine_distance NArray[[0,0,0],[1,2,3]], NArray[3,5,7]
    assert_equal 0.0, d[0]
    assert_in_delta 0.0026, d[1]
  end

    def test_multi_narray_with_zero2
    d = cosine_distance NArray[[1,2,3], [3,5,7]], NArray[0,0,0]
    p d
    assert_equal 0.0, d[0]
    assert_equal 0.0, d[1]
  end

  def test_with_narray_multirow
    d = cosine_distance NArray[[1,2,3], [2,3,4]], NArray[3,5,7]
    assert_in_delta 0.0026, d[0]
    assert_in_delta 0.0012, d[1]
  end

  def test_with_nmatrix
    d = cosine_distance NMatrix[[1.0,2.0,3.0], [2.0,3.0,4.0]], NMatrix[[3.0,5.0,7.0]]
    assert_in_delta 0.0026, d[0]
    assert_in_delta 0.0012, d[1]
  end

end

