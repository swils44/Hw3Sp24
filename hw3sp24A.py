def is_symmetric_positive_definite(A):
    n = len(A)
    # Check for symmetry
    for i in range(n):
        for j in range(i + 1, n):
            if A[i][j] != A[j][i]:
                return False

    # Check for positive definiteness
    try:
        cholesky_decomposition(A)
        return True
    except ValueError:
        return False


def cholesky_decomposition(A):
    n = len(A)
    L = [[0.0] * n for _ in range(n)]

    for i in range(n):
        for j in range(i + 1):
            sum_val = A[i][j]
            for k in range(j):
                sum_val -= L[i][k] * L[j][k]
            if i == j:
                if sum_val <= 0:
                    raise ValueError("Matrix not positive definite.")
                L[i][j] = (sum_val ** 0.5)
            else:
                L[i][j] = sum_val / L[j][j]

    return L


def cholesky_solve(A, b):
    L = cholesky_decomposition(A)

    # Solve Ly = b using forward substitution
    n = len(b)
    y = [0.0] * n
    for i in range(n):
        y[i] = b[i]
        for j in range(i):
            y[i] -= L[i][j] * y[j]
        y[i] /= L[i][i]

    # Solve L^T x = y using backward substitution
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = y[i]
        for j in range(i + 1, n):
            x[i] -= L[j][i] * x[j]
        x[i] /= L[i][i]

    return x


# Example matrices and vectors
A_cholesky = [[4, 2, -2], [2, 10, 4], [-2, 4, 11]]
A_doolittle = [[1, 2, 3], [2, 5, 2], [3, 2, 6]]
b = [4, 6, 3]

if is_symmetric_positive_definite(A_cholesky):
    solution = cholesky_solve(A_cholesky, b)
    method_used = "Cholesky"
else:
    solution = doolittle_solve(A_doolittle, b)
    method_used = "Doolittle"

print(f"Solution using {method_used} method: {solution}")
