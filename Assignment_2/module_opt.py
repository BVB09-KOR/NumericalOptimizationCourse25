############################################# Numerical Optimization 모듈(완성본 모음) #############################################
import numpy as np

######################### Central Difference Method - Scalar func / n-dim point x
### scipy.differentiate.derivative(f, x, ...) 함수 사용 가능
def grad_centraldiff(func, x):
    dim_x = x.shape[0]
    dfdx = np.empty([dim_x, 1])
    h = 1e-6
    for i in np.arange(dim_x):
        dx = np.zeros([dim_x, 1]); dx[i] = 1
        dfdx[i] = (func(x + h*dx) - func(x - h*dx))/(2*h)
    return dfdx

######################### Central Difference Method로 함수의 근사 hessian 계산하는 함수
### scipy.differentiate.hessian(f, x, ...) 함수 사용 가능
def hessian_centraldiff(func, x):
    n = len(x)
    H = np.zeros((n, n))
    h = 1e-5
    for i in range(n):      
        for j in range(n):
            x_ij_plus_plus = np.array(x, dtype=float)
            x_ij_plus_minus = np.array(x, dtype=float)
            x_ij_minus_plus = np.array(x, dtype=float)
            x_ij_minus_minus = np.array(x, dtype=float)
            
            x_ij_plus_plus[i] += h
            x_ij_plus_plus[j] += h
            
            x_ij_plus_minus[i] += h
            x_ij_plus_minus[j] -= h
            
            x_ij_minus_plus[i] -= h
            x_ij_minus_plus[j] += h
            
            x_ij_minus_minus[i] -= h
            x_ij_minus_minus[j] -= h
            
            H[i, j] = (func(x_ij_plus_plus) - func(x_ij_plus_minus) - func(x_ij_minus_plus) + func(x_ij_minus_minus)) / (4 * h**2)
    
    return H

######################### Search direction using Steepest Descent Method - Scalar func / n-dim point x
def search_direction_stp_descent(func, x):
    p = -grad_centraldiff(func, x)
    return p

######################### Search direction using Conjugate Gradient Method - Hestenes Stiefel Formula supplemented by Steepest Descent Method for numerical stability
######## Nonlinear CG 중 Hestenes-Stiefel algorithm 사용 시 beta 계산에서 분모가 0 되면 폭주 가능.
######## 따라서 Steepest descent랑 섞어서 그런 부분 방지해야 함.
### scipy.optimize.minimize(..., method='CG', ...) 함수 사용 가능
def search_direction_cg_hs(k, grad_old, grad_cur, p_old):
    if k == 0:
        p = -grad_cur
    else:
        num = (grad_cur - grad_old).T@grad_cur
        den = (grad_cur - grad_old).T@p_old
        if abs(den) < 1e-12 or np.isnan(den) or np.isnan(num): # 분모(den)가 0에 가까워지면 beta, p, x_new가 차례로 폭주하는 걸 방지하기 위해 이 경우 steepest descent method를 대신 사용.
            p = -grad_cur
        else:
            beta = num/den
            p = -grad_cur + beta*p_old
    return p

######################### Search direction using Conjugate Gradient Method - Fletcher-Reeves Formula
######## Nonlinear CG 중 Fletcher-Reeves algorithm 사용 시 beta 계산에서 분모가 0 될 일이 없기에 폭주 가능성 없음.
######## 따라서 Steepest descent랑 섞어서 안 써도 됨.
### scipy.optimize.minimize(..., method='CG', ...) 함수 사용 가능
def search_direction_cg_fr(k, grad_old, grad_cur, p_old):
    if k == 0:
        p = -grad_cur
    else:
        num = grad_cur.T@grad_cur
        den = grad_old.T@grad_old

        ######## steepest descent 부분 무효(주석)처리 ########
        # if abs(den) < 1e-12 or np.isnan(den) or np.isnan(num): # 분모(den)가 0에 가까워지면 beta, p, x_new가 차례로 폭주하는 걸 방지하기 위해 이 경우 steepest descent method를 대신 사용.
        #     p = -grad_cur
        # else:
        #     beta = num/den
        #     p = -grad_cur + beta*p_old

        beta = num/den
        p = -grad_cur + beta*p_old
        
    return p

######################### Search direction using Newton's Method
######## Hessian이 PD가 아니거나, x0가 x*에서 너무 먼 경우 수렴 보장 X.
def search_direction_newton(grad, hessian):
    p = -hessian@grad
    return p

######################### Search direction using Conjugate Gradient Method - Fletcher-Reeves Formula
### scipy.optimize.minimize(..., method='BFGS', ...) 함수 사용 가능
def search_direction_quasi_newton_bfgs(k, x_old, x_cur, grad_old, grad_cur, hessian_inv_aprx_old):
    dim_x = x_old.shape[0]
    if k == 0:
        hessian_inv_aprx = np.eye(dim_x)
    else:
        dx = x_cur - x_old
        dg = grad_cur - grad_old
        dgdx = float(dg.T @ dx)
        if abs(dgdx) < 1e-10:  # to avoid division by zero
            hessian_inv_aprx = np.eye(dim_x)
        else:
            I = np.eye(dim_x)
            rho = 1.0 / dgdx
            V = I - rho * dx @ dg.T
            hessian_inv_aprx = V @ hessian_inv_aprx_old @ V.T + rho * dx @ dx.T # BFGS formula

    p = -hessian_inv_aprx @ grad_cur
    return p, hessian_inv_aprx

######################### Backtracking algorithm - Scalar func / n-dim point x / n-dim grad_x / n-dim search direction p / curruent iteration k
# backtracking 알고리즘, 더 포괄적으로 step size alpha를 찾는 line search algorithm은 반드시 함수가 명시적으로 주어져야 한다.
# alpha를 찾기 위해서는 every alpha_try에서 function evaluation을 거쳐야 하기 때문이다.
def backtracking(func, x, grad_x, p, k):
    c1 = 1e-4
    c2 = 0.5

    alpha_try = 1
    x_try = x + alpha_try*p

    i = 0
    while func(x_try) > (func(x) + c1*alpha_try*grad_x.T@p):
        i = i + 1
        alpha_try = c2*alpha_try
        x_try = x + alpha_try*p
    alpha = alpha_try
    print(f'alpha_{k}_{i} = {alpha}\n')
    return alpha

def stp_descent(func, x_cur, tol):
    if type(x_cur) != np.ndarray:
        x_cur = np.array(x_cur)
    else:
        pass
    x_cur = x_cur.reshape(-1, 1)
    dim_x = x_cur.shape[0]

    print(f'x0 : {x_cur.reshape(dim_x)}') # 이렇게 메시지 출력할 때만 vector form으로 쓰자(메시지는 알아보기 쉬워야 하니까).
    grad_cur = grad_centraldiff(func, x_cur)
    k = 0

    #################################### NC for optimality check of initial guess ####################################
    if np.linalg.norm(x_cur) < tol:
        print(f'norm of grad at x0 : {np.linalg.norm(grad_cur)} --> x0 is optimum point !')
        x_new = x_cur
    else:
        print(f'norm of grad at x0 : {np.linalg.norm(grad_cur)} --> x0 is not optimum point. Optimization begins ...')

        #################################### Line search ####################################
        x_new = x_cur
        grad_new = grad_cur
        while np.linalg.norm(grad_new) > tol: #### Convergence Check ####
            #################################### Search direction p_cur ####################################
            x_cur = x_new
            grad_cur = grad_new
            p_cur = search_direction_stp_descent(func, x_cur) #### Steepest descent method ####
            print(f'p_{k} = {p_cur.reshape(dim_x)}')

            #################################### Step length alpha ####################################
            alpha = backtracking(func, x_cur, grad_cur, p_cur, k) #### backtracking algorithm ####

            # New point x_new
            x_new = x_cur + alpha*p_cur
            grad_new = grad_centraldiff(func, x_new)
            k = k + 1
            print(f'x_{k} = {x_new.reshape(dim_x)} / |grad(x_{k})| = {np.linalg.norm(grad_new)}')

    #################################### Complete Optimization ####################################
    print(f'optimization converges --> x* = {x_new.reshape(dim_x)} / |grad(x*)| = {np.linalg.norm(grad_new)}')
    return x_new

def cg_hs(func, x_cur, tol):
    if type(x_cur) != np.ndarray:
        x_cur = np.array(x_cur)
    else:
        pass    
    x_cur = x_cur.reshape(-1, 1)
    dim_x = x_cur.shape[0] # design space dimension

    print(f'x0 : {x_cur.reshape(dim_x)}') # 이렇게 메시지 출력할 때만 vector form으로 쓰자(메시지는 알아보기 쉬워야 하니까).
    grad_cur = grad_centraldiff(func, x_cur) # gradient of x0
    p_cur = -grad_cur # Search direction Initializion for 1st line search(to be used as p_old)
    k = 0 # Iteration initialization

    #################################### NC for optimality check of initial guess ####################################
    if np.linalg.norm(grad_cur) < tol:
        print(f'norm of grad at x0 : {np.linalg.norm(grad_cur)} --> x0 is optimum point !')
        x_new = x_cur
    else:
        print(f'norm of grad at x0 : {np.linalg.norm(grad_cur)} --> x0 is not optimum point. Optimization begins ...')

        #################################### Line search ####################################
        x_new = x_cur # Iteration loop 위해 이름 변경
        grad_new = grad_cur # Iteration loop 위해 이름 변경
        while np.linalg.norm(grad_new) > tol: #### Convergence Check ####
            #################################### Search direction p_cur ####################################
            grad_old = grad_cur
            p_old = p_cur

            x_cur = x_new
            grad_cur = grad_new

            p_cur = search_direction_cg_hs(k, grad_old, grad_cur, p_old)
            print(f'p_{k} = {p_cur.reshape(dim_x)}')

            #################################### Step length alpha ####################################
            alpha = backtracking(func, x_cur, grad_cur, p_cur, k) #### backtracking algorithm ####

            # New point x_new
            x_new = x_cur + alpha*p_cur
            grad_new = grad_centraldiff(func, x_new)
            k = k + 1
            print(f'x_{k} = {x_new.reshape(dim_x)} / |grad(x_{k})| = {np.linalg.norm(grad_new)}')

    #################################### Complete Optimization ####################################
    print(f'optimization converges --> x* = {x_new.reshape(dim_x)} / |grad(x*)| = {np.linalg.norm(grad_new)}')
    return x_new

def cg_fr(func, x_cur, tol):
    if type(x_cur) != np.ndarray:
        x_cur = np.array(x_cur)
    else:
        pass
    x_cur = x_cur.reshape(-1, 1)
    dim_x = x_cur.shape[0] # design space dimension

    print(f'x0 : {x_cur.reshape(dim_x)}') # 이렇게 메시지 출력할 때만 vector form으로 쓰자(메시지는 알아보기 쉬워야 하니까).
    grad_cur = grad_centraldiff(func, x_cur) # gradient of x0
    p_cur = -grad_cur # Search direction Initializion for 1st line search(to be used as p_old)
    k = 0 # Iteration initialization

    #################################### NC for optimality check of initial guess ####################################
    if np.linalg.norm(grad_cur) < tol:
        print(f'norm of grad at x0 : {np.linalg.norm(grad_cur)} --> x0 is optimum point !')
        x_new = x_cur
    else:
        print(f'norm of grad at x0 : {np.linalg.norm(grad_cur)} --> x0 is not optimum point. Optimization begins ...')

        #################################### Line search ####################################
        x_new = x_cur # Iteration loop 위해 이름 변경
        grad_new = grad_cur # Iteration loop 위해 이름 변경
        while np.linalg.norm(grad_new) > tol: #### Convergence Check ####
            #################################### Search direction p_cur ####################################
            grad_old = grad_cur
            p_old = p_cur

            x_cur = x_new
            grad_cur = grad_new

            p_cur = search_direction_cg_fr(k, grad_old, grad_cur, p_old)
            print(f'p_{k} = {p_cur.reshape(dim_x)}')

            #################################### Step length alpha ####################################
            alpha = backtracking(func, x_cur, grad_cur, p_cur, k) #### backtracking algorithm ####

            # New point x_new
            x_new = x_cur + alpha*p_cur
            grad_new = grad_centraldiff(func, x_new)
            k = k + 1
            print(f'x_{k} = {x_new.reshape(dim_x)} / |grad(x_{k})| = {np.linalg.norm(grad_new)}')

    #################################### Complete Optimization ####################################
    print(f'optimization converges --> x* = {x_new.reshape(dim_x)} / |grad(x*)| = {np.linalg.norm(grad_new)}')
    return x_new

def newton(func, x_cur, tol):
    if type(x_cur) != np.ndarray:
        x_cur = np.array(x_cur)
    else:
        pass
    x_cur = x_cur.reshape(-1, 1)
    dim_x = x_cur.shape[0] # design space dimension

    print(f'x0 : {x_cur.reshape(dim_x)}') # 이렇게 메시지 출력할 때만 vector form으로 쓰자(메시지는 알아보기 쉬워야 하니까).
    grad_cur = grad_centraldiff(func, x_cur) # gradient of x0
    k = 0 # Iteration initialization

    #################################### NC for optimality check of initial guess ####################################
    if np.linalg.norm(grad_cur) < tol:
        print(f'norm of grad at x0 : {np.linalg.norm(grad_cur)} --> x0 is optimum point !')
        x_new = x_cur
    else:
        print(f'norm of grad at x0 : {np.linalg.norm(grad_cur)} --> x0 is not optimum point. Optimization begins ...')

        #################################### Line search ####################################
        x_new = x_cur # Iteration loop 위해 이름 변경
        grad_new = grad_cur # Iteration loop 위해 이름 변경
        while np.linalg.norm(grad_new) > tol: #### Convergence Check ####
            #################################### Search direction p_cur ####################################
            x_cur = x_new
            grad_cur = grad_new
            hessian_cur = hessian_centraldiff(func, x_cur)

            p_cur = search_direction_newton(grad_cur, hessian_cur)
            print(f'p_{k} = {p_cur.reshape(dim_x)}')

            #################################### Step length alpha ####################################
            alpha = backtracking(func, x_cur, grad_cur, p_cur, k) #### backtracking algorithm ####

            # New point x_new
            x_new = x_cur + alpha*p_cur
            grad_new = grad_centraldiff(func, x_new)
            k = k + 1
            print(f'x_{k} = {x_new.reshape(dim_x)} / |grad(x_{k})| = {np.linalg.norm(grad_new)}')

    #################################### Complete Optimization ####################################
    print(f'optimization converges --> x* = {x_new.reshape(dim_x)} / |grad(x*)| = {np.linalg.norm(grad_new)}')

def quasi_newton_bfgs(func, x_cur, tol):
    if type(x_cur) != np.ndarray:
        x_cur = np.array(x_cur)
    else:
        pass
    x_cur = x_cur.reshape(-1, 1)
    dim_x = x_cur.shape[0] # design space dimension

    print(f'x0 : {x_cur.reshape(dim_x)}') # 이렇게 메시지 출력할 때만 vector form으로 쓰자(메시지는 알아보기 쉬워야 하니까).
    grad_cur = grad_centraldiff(func, x_cur) # gradient of x0
    k = 0 # Iteration initialization

    #################################### NC for optimality check of initial guess ####################################
    if np.linalg.norm(grad_cur) < tol:
        print(f'norm of grad at x0 : {np.linalg.norm(grad_cur)} --> x0 is optimum point !')
        x_new = x_cur
    else:
        print(f'norm of grad at x0 : {np.linalg.norm(grad_cur)} --> x0 is not optimum point. Optimization begins ...')

        #################################### Line search ####################################
        x_new = x_cur # Iteration loop 위해 이름 변경
        grad_new = grad_cur # Iteration loop 위해 이름 변경
        hessian_inv_aprx_cur = np.identity(dim_x) # Iteration loop 위해 initialization
        while np.linalg.norm(grad_new) > tol: #### Convergence Check ####
            #################################### Search direction p_cur ####################################
            x_old = x_cur
            x_cur = x_new
            grad_old = grad_cur
            grad_cur = grad_new
            hessian_inv_aprx_old = hessian_inv_aprx_cur

            p_cur, hessian_inv_aprx_cur = search_direction_quasi_newton_bfgs(k, x_old, x_cur, grad_old, grad_cur, hessian_inv_aprx_old)
            print(f'p_{k} = {p_cur.reshape(dim_x)}')

            #################################### Step length alpha ####################################
            alpha = backtracking(func, x_cur, grad_cur, p_cur, k) #### backtracking algorithm ####

            # New point x_new
            x_new = x_cur + alpha*p_cur
            grad_new = grad_centraldiff(func, x_new)
            k = k + 1
            print(f'x_{k} = {x_new.reshape(dim_x)} / |grad(x_{k})| = {np.linalg.norm(grad_new)}')

    #################################### Complete Optimization ####################################
    print(f'optimization converges --> x* = {x_new.reshape(dim_x)} / |grad(x*)| = {np.linalg.norm(grad_new)}')
    return x_new