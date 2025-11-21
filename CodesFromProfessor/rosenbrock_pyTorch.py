
import torch

# Define the Rosenbrock function
def rosenbrock(x):
    return torch.sum(100.0 * (x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0)

# Create a tensor with gradient tracking
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

# Compute the function value
y = rosenbrock(x)

# Compute the gradient
y.backward()

# Print the gradient
print("Gradient of the Rosenbrock function at", x.detach().numpy(), "is", x.grad.numpy())