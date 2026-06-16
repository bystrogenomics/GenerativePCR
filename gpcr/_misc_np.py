import numpy as np


def softplus_inverse_np(y):
    """
    Compute the inverse of the softplus function in a numerically stable way.

    Softplus is defined as ``y = log(exp(x) + 1)``, so its inverse is
    ``x = log(exp(y) - 1)``. For very small or very large ``y``, the naive
    formula is numerically unstable; this function handles those edge cases
    by clamping and substitution.

    Parameters
    ----------
    y : array-like
        Input array; should contain positive values (the range of softplus).

    Returns
    -------
    x : ndarray
        Array of the same shape as ``y`` containing the inverse softplus
        values.
    """
    min_threshold = 10**-15
    max_threshold = 500
    safe_y = np.clip(
        y, min_threshold, max_threshold
    )  # we can safely pass this to the reference inverse_softplus below
    safe_x = np.log(np.exp(safe_y) - 1)

    # if y_i was below (respectively: above) the min (max) threshold, replace with log(y_i)  (y_i)
    x = np.where(y < min_threshold, np.log(y), safe_x)
    x = np.where(y > max_threshold, y, x)
    return x


def subset_square_matrix_np(Sigma, idxs):
    """
    Extract a symmetric submatrix from a square matrix.

    Parameters
    ----------
    Sigma : ndarray of shape (n, n)
        The square matrix to subset (typically a covariance matrix).

    idxs : ndarray of shape (n,)
        Boolean or integer index array where 1 marks the rows/columns
        to select.

    Returns
    -------
    Sigma_sub : ndarray of shape (k, k)
        The submatrix indexed by ``idxs == 1``, where ``k = idxs.sum()``.
    """
    Sigma_sub = Sigma[np.ix_(idxs == 1, idxs == 1)]
    return Sigma_sub


def classify_missingness(matrix):
    """
    Group rows of a matrix by their NaN-missingness pattern.

    Identifies unique patterns of missing values (NaN positions) across
    rows and groups the rows accordingly.

    Parameters
    ----------
    matrix : ndarray of shape (n_samples, n_features)
        Input matrix potentially containing NaN values.

    Returns
    -------
    matrices_list : list of ndarray
        List of sub-matrices, one per unique missingness pattern.
        Each sub-matrix contains all rows sharing that pattern.

    vectors_list : list of ndarray of bool
        List of boolean observation masks, one per unique pattern,
        where ``True`` indicates that the feature is observed
        (not NaN) in that group.
    """
    pattern_dict = {}
    matrices_list = []
    vectors_list = []

    for i, row in enumerate(matrix):
        # Using a hashable representation of the NaN pattern
        pattern = tuple(np.isnan(row))

        if pattern not in pattern_dict:
            pattern_dict[pattern] = [i]
        else:
            pattern_dict[pattern].append(i)

    for indices in pattern_dict.values():
        sub_matrix = matrix[indices]
        matrices_list.append(sub_matrix)
        vectors_list.append(~np.isnan(sub_matrix[0]))

    return matrices_list, vectors_list
