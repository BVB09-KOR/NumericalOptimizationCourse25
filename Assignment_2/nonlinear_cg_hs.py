all = [var for var in globals() if var[0] != '_']
for var in all:
    del var
del all

import numpy as np
from module_opt import *

# Target function
func_rosenbrock = lambda x : (1.0 - x[0])**2 + 100*(x[1] - x[0]**2)**2

#################################### Define optimization problem  ####################################
obj = func_rosenbrock

#################################### Tolerance setting ####################################
tol = 1e-7

#################################### Initial guess ####################################
x_cur = np.array([[0], 
                  [3]]) # 2 x 1 matrix(※ matlab과 다른 점. numpyp는 vector를 1차원 array, matrix를 2차원 array로 엄밀하게 구분해서 다룬다.)
                      # 따라서 웬만하면 그냥 vector도 matrix form으로 쓰는 걸 추천. 나중에 matrix와 vector를 함께 계산할 일이 많기 때문.
dim_x = x_cur.shape[0] # design space dimension

print(f'x0 : {x_cur.reshape(dim_x)}') # 이렇게 메시지 출력할 때만 vector form으로 쓰자(메시지는 알아보기 쉬워야 하니까).
grad_cur = grad_centraldiff(obj, x_cur) # gradient of x0
p_cur = -grad_cur # Search direction Initializion for 1st line search(to be used as p_old)
k = 0 # Iteration initialization

#################################### NC for optimality check of initial guess ####################################
if np.linalg.norm(grad_cur) < tol:
    print(f'norm of grad at x0 : {np.linalg.norm(grad_cur)} --> x0 is optimum point !')
else:
    print(f'norm of grad at x0 : {np.linalg.norm(grad_cur)} --> x0 is not optimum point. Optimization begins ...')

    #################################### Line search ####################################
    x_new = x_cur # Iteration loop 위해 이름 변경
    grad_new = grad_cur # Iteration loop 위해 이름 변경
    while np.linalg.norm(grad_new) > tol: #### Convergence Check ####
        #################################### Search direction p_cur ####################################
        x_old = x_cur
        grad_old = grad_cur
        p_old = p_cur

        x_cur = x_new
        grad_cur = grad_new

        p_cur = cg_nl_hs(k, grad_old, grad_cur, p_old)
        print(f'p_{k} = {p_cur.reshape(dim_x)}')

        #################################### Step length alpha ####################################
        alpha = backtracking(obj, x_cur, grad_cur, p_cur, k) #### backtracking algorithm ####

        # New point x_new
        x_new = x_cur + alpha*p_cur
        grad_new = grad_centraldiff(obj, x_new)
        k = k + 1
        print(f'x_{k} = {x_new.reshape(dim_x)} / |grad(x_{k})| = {np.linalg.norm(grad_new)}')

#################################### Complete Optimization ####################################
print(f'optimization converges --> x* = {x_new.reshape(dim_x)} / |grad(x*)| = {np.linalg.norm(grad_new)}')
