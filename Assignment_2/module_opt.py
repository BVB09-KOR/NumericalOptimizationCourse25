############################################# Numerical Optimization 모듈 #############################################
import numpy as np

######################### Central Difference Method - Scalar func / n-dim point x
def grad_centraldiff(func, x):
    dim_x = x.shape[0]
    dfdx = np.empty([dim_x, 1])
    h = 1e-6
    for i in np.arange(dim_x):
        dx = np.zeros([dim_x, 1]); dx[i] = 1
        dfdx[i] = (func(x + h*dx) - func(x - h*dx))/(2*h)
    return dfdx

######################### Steepest Descent Method - Scalar func / n-dim point x
def stp_descent(func, x):
    p = -grad_centraldiff(func, x)
    return p

######################### Conjugate Gradient Method - Hestenes Stiefel Formula supplemented by Steepest Descent Method for numerical stability
def cg_nl_hs(k, grad_old, grad_cur, p_old):
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