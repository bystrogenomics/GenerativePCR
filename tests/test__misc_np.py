import numpy as np

from gpcr._misc_np import (
    classify_missingness,
    softplus_inverse_np,
    subset_square_matrix_np,
)

# ---------------------------------------------------------
# Tests for softplus_inverse_np
# ---------------------------------------------------------


def test_softplus_inverse_np_normal_values():
    """Test the inverse softplus function on typical intermediate values."""
    # Start with original x values
    x_orig = np.array([-2.0, 0.0, 2.0, 5.0])
    # Apply softplus: y = log(exp(x) + 1)
    y = np.log(np.exp(x_orig) + 1)
    # Recover x
    x_recovered = softplus_inverse_np(y)

    np.testing.assert_allclose(x_recovered, x_orig, rtol=1e-5)


def test_softplus_inverse_np_below_min_threshold():
    """Test values falling below the min_threshold (10**-15)."""
    # For very small y, x should simply be log(y)
    y = np.array([1e-16, 1e-20])
    expected_x = np.log(y)

    result = softplus_inverse_np(y)
    np.testing.assert_allclose(result, expected_x)


def test_softplus_inverse_np_above_max_threshold():
    """Test values falling above the max_threshold (500)."""
    # For very large y, x should evaluate to y
    y = np.array([501.0, 1000.0])
    expected_x = y

    result = softplus_inverse_np(y)
    np.testing.assert_allclose(result, expected_x)


# ---------------------------------------------------------
# Tests for subset_square_matrix_np
# ---------------------------------------------------------


def test_subset_square_matrix_np_standard():
    """Test subsetting a matrix with a mix of 1s and 0s."""
    Sigma = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    # Select the first and third rows/columns
    idxs = np.array([1, 0, 1])

    expected_sub = np.array([[1, 3], [7, 9]])

    result = subset_square_matrix_np(Sigma, idxs)
    np.testing.assert_array_equal(result, expected_sub)


def test_subset_square_matrix_np_all_selected():
    """Test subsetting when all indices are selected."""
    Sigma = np.eye(3)
    idxs = np.array([1, 1, 1])

    result = subset_square_matrix_np(Sigma, idxs)
    np.testing.assert_array_equal(result, Sigma)


def test_subset_square_matrix_np_none_selected():
    """Test subsetting when no indices are selected."""
    Sigma = np.eye(3)
    idxs = np.array([0, 0, 0])

    result = subset_square_matrix_np(Sigma, idxs)
    assert result.shape == (0, 0)
    assert result.size == 0


# ---------------------------------------------------------
# Tests for classify_missingness
# ---------------------------------------------------------


def test_classify_missingness_mixed_nans():
    """Test classifying missingness with various NaN patterns."""
    matrix = np.array(
        [
            [1.0, np.nan, 3.0],  # Pattern A
            [4.0, 5.0, 6.0],  # Pattern B (No NaNs)
            [7.0, np.nan, 9.0],  # Pattern A
            [np.nan, 11.0, 12.0],  # Pattern C
        ]
    )

    matrices_list, vectors_list = classify_missingness(matrix)

    # We should have 3 distinct patterns
    assert len(matrices_list) == 3
    assert len(vectors_list) == 3

    # Because dictionary iteration order guarantees are Python-version dependent,
    # we dynamically check the properties of the returned lists instead of relying on exact index order.
    shapes = [m.shape for m in matrices_list]
    assert (2, 3) in shapes  # One pattern has 2 rows
    assert shapes.count((1, 3)) == 2  # Two patterns have 1 row

    # Verify the contents of each pattern group
    for m, v in zip(matrices_list, vectors_list):
        if m.shape[0] == 2:
            # This is Pattern A (Rows 0 and 2)
            np.testing.assert_array_equal(v, [True, False, True])
            np.testing.assert_allclose(m[0, [0, 2]], [1.0, 3.0])
            np.testing.assert_allclose(m[1, [0, 2]], [7.0, 9.0])
        elif np.isnan(m[0, 0]):
            # This is Pattern C (Row 3)
            np.testing.assert_array_equal(v, [False, True, True])
            np.testing.assert_allclose(m[0, [1, 2]], [11.0, 12.0])
        else:
            # This is Pattern B (Row 1)
            np.testing.assert_array_equal(v, [True, True, True])
            np.testing.assert_allclose(m[0], [4.0, 5.0, 6.0])


def test_classify_missingness_no_nans():
    """Test classifying a matrix with absolutely no missing values."""
    matrix = np.ones((4, 3))

    matrices_list, vectors_list = classify_missingness(matrix)

    # Should only return one group
    assert len(matrices_list) == 1
    assert len(vectors_list) == 1

    np.testing.assert_array_equal(matrices_list[0], matrix)
    np.testing.assert_array_equal(vectors_list[0], [True, True, True])
