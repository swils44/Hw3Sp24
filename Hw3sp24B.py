import math

def t_distribution_probability(degrees_of_freedom, z_value):
    if degrees_of_freedom <= 0:
        raise ValueError("Degrees of freedom must be greater than 0.")

    # Use cumulative distribution function (cdf) for t-distribution
    x = z_value / math.sqrt(degrees_of_freedom)
    probability = 0.5 * (1 + math.erf(x))

    return probability

def main():
    z_values = [1.645, 1.96, 2.576]

    for degrees_of_freedom in [7, 11, 15]:
        print(f"\nDegrees of Freedom: {degrees_of_freedom}")
        for z_value in z_values:
            probability = t_distribution_probability(degrees_of_freedom, z_value)
            print(f"Probability for z={z_value}: {probability}")

if __name__ == "__main__":
    main()
