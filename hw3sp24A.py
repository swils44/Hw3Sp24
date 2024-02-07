from hw2c import doolittle_solve as ds
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

def main():
    # Example matrices and vectors
    MatA = [[1, -1, 3, 2], [-1, 5, -5, -2], [3, -5, 19, 3], [2, -2, 3, 21]]
    MatB = [[4, 2, 4, 0], [2, 2, 3, 2], [4, 3, 6, 3], [0, 2, 3, 9]]
    b = [15, -35, 94, 1]
    b2 = [20, 36, 60, 122]
    if is_symmetric_positive_definite(MatA) and is_symmetric_positive_definite(MatB):
        solution = cholesky_solve(MatA, b)
        solution2 = cholesky_solve(MatB, b2)
        method_used = "Cholesky for both"
        print(f"Used {method_used} systems. System 1: {solution} System 2 {solution2}")

    elif is_symmetric_positive_definite(MatB):
        solution = cholesky_solve(MatB, b2)
        solution2 = ds(MatA, b)
        method_used = "Cholesky for system 2 & Doolittle for system 1"
        print(f"Used {method_used}: System 2: {solution2} System 1: {solution}")

    elif is_symmetric_positive_definite(MatA):
        solution = cholesky_solve(MatA, b)
        solution2 = ds(MatB, b2)
        method_used = "Cholesky for system 1 & Doolittle for system 2"
        print(f"Used {method_used}: System1: {solution} System 2: {solution}")

    else:
        solution = ds(MatA, b)
        solution2 = ds(MatB, b2)
        method_used = "Doolittle for both"
        print(f"Solution using {method_used}: System 1:{solution} System 2: {solution2} ")

if __name__ == "__main__":
    main()