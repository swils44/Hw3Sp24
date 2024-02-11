import math

def Simpson(f, a, b, n=1000):
    """
    Approximate the definite integral of a function using Simpson's rule.

    Parameters:
        f (function): The function to integrate.
        a (float): The lower limit of integration.
        b (float): The upper limit of integration.
        n (int, optional): Number of subintervals (even). Defaults to 1000.

    Returns:
        float: Approximated integral value.
    """
    if n % 2 == 1:
        raise ValueError("n must be even")

    # Calculate the step size
    h = (b - a) / n
    # Create a list of x values for each subinterval
    x = [a + i * h for i in range(n + 1)]
    # Initialize integral with the values at the endpoints
    integral = f(a) + f(b)

    # Apply Simpson's rule
    for i in range(1, n, 2):
        integral += 4 * f(x[i])

    for i in range(2, n - 1, 2):
        integral += 2 * f(x[i])

    integral *= h / 3
    return integral


def tdp(m, z):
    """
    Calculate the probability for a given t-distribution and z-value.

    Parameters:
        m (float): Degrees of freedom for the t-distribution.
        z (float): Z-value for which probability is calculated.

    Returns:
        float: Probability value.
    """
    if m <= 0:
        raise ValueError("Degrees of freedom (m) must be greater than zero")

    # Define the t-distribution function
    def tdf(x):
        return (math.gamma((0.5 * m) + 0.5) / (math.sqrt(m * math.pi) * math.gamma(m / 2)) *
                (1 + ((x ** 2) / m)) ** (-(m + 1) / 2))

    # Set a lower limit for integration (10 times the z-value)
    lower = z - (10 * z)
    # Use Simpson's rule to calculate the probability
    probability = Simpson(tdf, lower, z, 1000)
    return probability


def main():
    """
    Main function to get user input and calculate probabilities.
    """
    df = []
    z_val = []

    # Get degrees of freedom input
    for i in range(3):
        df.append(int(input(f"Enter value of degrees of freedom #({i + 1}) (Integer): ")))

    # Get z-values input
    for j in range(3):
        z_val.append(float(input(f"Enter z {j + 1}: ")))

    # Calculate and display probabilities for each combination of degrees of freedom and z-value
    for k in range(3):
        for l in range(3):
            probability = tdp(df[k], z_val[l])
            print(f"The probability for {df[k]} degrees of freedom and a z value of {z_val[l]} is {probability:.4f}")


if __name__ == "__main__":
    main()
