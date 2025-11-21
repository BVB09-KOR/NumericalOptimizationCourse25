import jax # JAX is the AD library
import jax.numpy # Load JAX's version of the NumPy library

# Define the Rosenbrock function
def rosenbrock(x):
    return jax.numpy.sum(100.0 * (x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0)

# Compute the gradient using JAX's automatic differentiation
rosenbrock_grad = jax.grad(rosenbrock)

# Example input
x = jax.numpy.array([1.0, 2.0, 3.0])
gradient = rosenbrock_grad(x)

# Print the gradient
print("Gradient of the Rosenbrock function at", x, "is", gradient)

