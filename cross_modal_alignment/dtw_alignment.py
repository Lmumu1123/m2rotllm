import numpy as np


def _zscore(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    m = float(np.mean(x))
    s = float(np.std(x))
    if s < eps:
        return x - m
    return (x - m) / s


def derivative(x: np.ndarray) -> np.ndarray:
    """
    First-order discrete derivative with same length (pads 0 at the end).
    """
    x = np.asarray(x, dtype=np.float32)
    d = np.diff(x, prepend=x[0])
    return d.astype(np.float32)


def constrained_dtw_path(
    x: np.ndarray,
    y: np.ndarray,
    *,
    w: int,
    alpha: float = 0.7,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Constrained DTW with Sakoe-Chiba band.

    Cost uses both values and derivatives:
      C(i,j) = alpha*(x[i]-y[j])^2 + (1-alpha)*(dx[i]-dy[j])^2

    Returns:
      path_x: indices in x along the warping path
      path_y: indices in y along the warping path
    """
    x = _zscore(x)
    y = _zscore(y)
    dx = derivative(x)
    dy = derivative(y)

    n = x.shape[0]
    m = y.shape[0]
    w = int(max(0, w))

    # DP arrays for banded computation.
    # We store only the band columns for each i.
    inf = np.float32(1e30)
    D = np.full((n, m), inf, dtype=np.float32)
    ptr = np.full((n, m), -1, dtype=np.int8)  # 0:diag, 1:up, 2:left

    D[0, 0] = 0.0

    i_start = 0
    i_end = n
    for i in range(i_start, i_end):
        jmin = max(0, i - w)
        jmax = min(m - 1, i + w)
        for j in range(jmin, jmax + 1):
            if i == 0 and j == 0:
                continue

            cost_val = x[i] - y[j]
            cost_dv = dx[i] - dy[j]
            cost = np.float32(alpha * cost_val * cost_val + (1.0 - alpha) * cost_dv * cost_dv)

            best = inf
            best_ptr = -1
            # diag
            if i > 0 and j > 0:
                v = D[i - 1, j - 1] + cost
                if v < best:
                    best = v
                    best_ptr = 0
            # up
            if i > 0:
                v = D[i - 1, j] + cost
                if v < best:
                    best = v
                    best_ptr = 1
            # left
            if j > 0:
                v = D[i, j - 1] + cost
                if v < best:
                    best = v
                    best_ptr = 2

            D[i, j] = best
            ptr[i, j] = best_ptr

    # backtrack
    i = n - 1
    j = m - 1
    if not np.isfinite(D[i, j]) or D[i, j] >= inf / 10:
        raise RuntimeError("DTW failed (path not found within band). Increase w.")

    path_x = []
    path_y = []
    while True:
        path_x.append(i)
        path_y.append(j)
        if i == 0 and j == 0:
            break
        p = int(ptr[i, j])
        if p == 0:
            i -= 1
            j -= 1
        elif p == 1:
            i -= 1
        elif p == 2:
            j -= 1
        else:
            raise RuntimeError("DTW backtrack got invalid pointer.")

    path_x.reverse()
    path_y.reverse()
    return np.array(path_x, dtype=np.int32), np.array(path_y, dtype=np.int32)


def best_lag_correlation(x: np.ndarray, y: np.ndarray, *, max_lag: int) -> int:
    """
    Find lag L (on y) that maximizes correlation with x.
    Convention:
      overlap: x[i] with y[i+L], when L>=0
      for L<0: x[i-L] with y[i]
    Returns:
      best L in [-max_lag, max_lag]
    """
    x = _zscore(x)
    y = _zscore(y)
    n = x.shape[0]
    m = y.shape[0]
    best_L = 0
    best_score = -1e30

    for L in range(-max_lag, max_lag + 1):
        if L >= 0:
            # y is shifted forward relative to x
            i0 = 0
            i1 = min(n, m - L)
            j0 = L
            j1 = L + i1
            if i1 <= 5:
                continue
            score = float(np.dot(x[i0:i1], y[j0:j1]))
        else:
            # L < 0, y is shifted backward
            j0 = 0
            j1 = min(m, n + L)
            i0 = -L
            i1 = i0 + j1
            if j1 <= 5:
                continue
            score = float(np.dot(x[i0:i1], y[j0:j1]))

        if score > best_score:
            best_score = score
            best_L = L

    return best_L
