import numpy
from scipy.optimize import minimize
import jax
import jax.numpy

# jax 내부 연산이 float64 데이터 타입 기반으로 작동되게 설정 ... 원래 jax 내부 연산 시 default 데이터타입은 float32임.
jax.config.update("jax_enable_x64", True)

# Objective function
def objective(x):
    return x[0]**2 + x[1]**2

# Gradient of the objective
def objective_grad(x):
    return numpy.array([2*x[0], 2*x[1]])

# Constraint: x^2 + y^2 - 1 = 0
def constraint(x):
    return x[0]**2 + x[1]**2 - 1

# Gradient of the constraint
def constraint_grad(x):
    return numpy.array([2*x[0], 2*x[1]])


# Compute gradients using JAX
objective_grad_j = jax.grad(objective)
constraint_grad_j = jax.grad(constraint)

# Convert JAX arrays to numPy for scipy
def objective_grad_jax(x):
    return numpy.array(objective_grad_j(x)).astype(numpy.float64)

def constraint_grad_jax(x):
    return numpy.array(constraint_grad_j(x)).astype(numpy.float64)

# Initial guess
x0 = numpy.array([0.5, 0.5])

# Assemble Constraint
cons = {'type': 'eq', 'fun': constraint}
cons_withgrad = {'type': 'eq', 'fun': constraint, 'jac': constraint_grad}
cons_withgrad_jax = {'type': 'eq', 'fun': constraint, 'jac': constraint_grad_jax}

# Minimize using SLSQP
minimize_function1 = minimize(objective, x0, method='SLSQP', constraints=[cons])
minimize_function2 = minimize(objective, x0, method='SLSQP', jac=objective_grad, constraints=[cons_withgrad])
minimize_function3 = minimize(objective, x0, method='SLSQP', jac=objective_grad_jax, constraints=[cons_withgrad_jax])


# Output - Analytical Gradient를 써서 구한 x*와 AD를 써서 구한 x*가 같은 것을 볼 수 있다.
print("Optimal solution (x, y):                              ", minimize_function1.x)
print("Optimal solution with Analytical Gradient(x, y):      ", minimize_function2.x) # Analytical Gradient 써서 구한 x*
print("Optimal solution with Automatic Differentiation(x, y):", minimize_function3.x) # AD 써서 구한 x*
