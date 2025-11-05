############################################# Numerical Optimization 모듈(완성본 모음) #############################################
import numpy as np
import io
import contextlib

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
######################### Derivative/Hessian Caculation #########################
### Central Difference Method based Gradient - Scalar func / n-dim point x
# scipy.differentiate.derivative(f, x, ...) 함수 사용 가능
def grad_centraldiff(f, x):
    x = np.atleast_1d(x)
    rel_step = 1e-6
    dfdx = np.zeros(len(x))
    for i in range(len(x)):
        h = rel_step*np.max([np.abs(x[i]), 1])
        e_unit = np.zeros(len(x)); e_unit[i] = 1
        dx = h*e_unit
        num = f(x+dx) - f(x-dx)
        den = 2*h
        dfdx[i] = num/den
    if not np.isfinite(dfdx).all(): # Check finitude
        raise ValueError('At least one component of gradient is not finite !')
    return dfdx

### Central Difference Method based Hessian
# scipy.differentiate.hessian(f, x, ...) 함수 사용 가능
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
    if not np.isfinite(H).all():
        print(f'Warning : Hessian approximation includes NaN !')
        # raise ValueError('Warning : Hessian approximation includes NaN !')
    return H

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
######################### Search direction algorithms #########################
### Search direction using Steepest Descent Method - Scalar func / n-dim point x
def search_direction_stp_descent(func, x):
    p = -grad_centraldiff(func, x)

    if not np.isfinite(p).all():
        raise ValueError("Non-finite number detected in search_direction_stp_descent (NaN or inf)")

    return p

### Search direction using Conjugate Gradient Method - Hestenes Stiefel(CGM-HS) Formula
# Nonlinear CG 중 Hestenes-Stiefel algorithm 사용 시 beta 계산에서 분모가 0 되면 폭주 가능.
# 따라서 Steepest descent랑 섞어서 그런 부분 방지해야 함.
# scipy.optimize.minimize(..., method='CG', ...) 함수 사용 가능
def search_direction_cg_hs(k, grad_old, grad_cur, p_old):
    if k == 0:
        p = -grad_cur
    else:
        num = (grad_cur - grad_old) @ grad_cur
        den = (grad_cur - grad_old) @ p_old
        if abs(den) < 1e-12 or np.isnan(den) or np.isnan(num): # 분모(den)가 0에 가까워지면 beta, p, x_new가 차례로 폭주하는 걸 방지하기 위해 이 경우 steepest descent method를 대신 사용.
            p = -grad_cur
        else:
            beta = num/den
            p = -grad_cur + beta*p_old
    
    if not np.isfinite(p).all():
        raise ValueError("Non-finite number detected in search_direction_cg_hs (NaN or inf)")

    return p

### Search direction using Conjugate Gradient Method - Fletcher Reeves(CGM-FR) Formula
# Nonlinear CG 중 Fletcher-Reeves algorithm 사용 시 beta 계산에서 분모가 0 될 일이 없기에 폭주 가능성 없음.
# 따라서 Steepest descent랑 섞어서 안 써도 됨. -> 수정 : grad_old가 0에 가까울 시 분모 0 될 수 있음.
# scipy.optimize.minimize(..., method='CG', ...) 함수 사용 가능
def search_direction_cg_fr(k, grad_old, grad_cur, p_old):
    if k == 0:
        p = -grad_cur
    else:
        num = grad_cur @ grad_cur
        den = grad_old @ grad_old

        ####### steepest descent 부분 무효(주석)처리 ########
        if abs(den) < 1e-12 or np.isnan(den) or np.isnan(num): # 분모(den)가 0에 가까워지면 beta, p, x_new가 차례로 폭주하는 걸 방지하기 위해 이 경우 steepest descent method를 대신 사용.
            p = -grad_cur
        else:
            beta = num/den
            p = -grad_cur + beta*p_old

        beta = num/den
        p = -grad_cur + beta*p_old

    if not np.isfinite(p).all():
        raise ValueError("Non-finite number detected in search_direction_cg_fr (NaN or inf)")
        
    return p

### Search direction using Newton's Method
# Hessian이 PD가 아니거나, x0가 x*에서 너무 먼 경우 수렴 보장 X.
def search_direction_newton(grad, hessian):
    p = -np.linalg.solve(hessian, grad)
    return p

### Search direction using Quasi-Newton Method - BFGS(QNM-BFGS) Formula
# scipy.optimize.minimize(..., method='BFGS', ...) 함수 사용 가능
def search_direction_quasi_newton_bfgs(k, x_old, x_cur, grad_old, grad_cur, hessian_inv_aprx_old):
    dim_x = len(x_cur)
    if k == 0: # 첫 iteration의 근사 Hessian inverse는 I로 설정
        hessian_inv_aprx = np.eye(dim_x)
    else: # 2번째 이후 iteration부터의 근사 Hessian inverse부터는 BFGS로 구함
        dx = x_cur - x_old
        dg = grad_cur - grad_old
        dgdx = dg @ dx
        if abs(dgdx) < 1e-10:  # to avoid division by zero
            hessian_inv_aprx = np.eye(dim_x)
        else:
            I = np.eye(dim_x)
            rho = 1.0 / dgdx
            V = I - rho * np.outer(dx, dg) # 벡터로 행렬 생성 연산은 @ 연산 쓰지 말고 대신 np.outer(a, b) 함수 써라.
            hessian_inv_aprx = V @ hessian_inv_aprx_old @ V.T + rho * np.outer(dx, dx) # BFGS formula

    p = -hessian_inv_aprx @ grad_cur

    if not np.isfinite(p).all():
        raise ValueError("Non-finite number detected in search_direction_quasi_newton_bfgs (NaN or inf)")

    return p, hessian_inv_aprx

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
######################### Step length search algorithms #########################
### Step length search A) Backtracking algorithm - Scalar func / n-dim point x / n-dim grad_x / n-dim search direction p / curruent iteration k
# backtracking 알고리즘, 더 포괄적으로 step size alpha를 찾는 line search algorithm은 반드시 함수가 명시적으로 주어져야 한다.
# alpha를 찾기 위해서는 every alpha_try에서 function evaluation을 거쳐야 하기 때문이다.
def backtracking(func, x, grad_x, p, k):
    c1 = 1e-4
    c2 = 0.5

    alpha_try = 1
    x_try = x + alpha_try*p

    i = 0
    
    print(f"func(x)={func(x)}, func(x_try)={func(x_try)}, grad·p={grad_x.T@p}")    

    while func(x_try) > (func(x) + c1*alpha_try*grad_x.T@p):
        i = i + 1
        alpha_try = c2*alpha_try
        x_try = x + alpha_try*p

    alpha = alpha_try

    print(f'alpha_{k}_{i} = {alpha}\n')

    return alpha

### Step length search B) Strong Wolfe's Conditions + Interpolation algorithm- Scalar func / n-dim point x / n-dim grad_x / n-dim search direction p / curruent iteration k
# interpol_alpha ⊂ bracketing_alpha ⊂ wolfe_strong_interpol
def interpol_alpha(f, x_cur, p_cur, a, b):
    # bracketing_alpha에서 계속해서 업데이트되는 구간(alpha_lo, alpha_hi)에 대해, 양단 점을 기반으로 2(3)차 함수로 근사하여 alpha_new 찾는 함수
    # 이 alpha_new는 bracketing_alpha에서 구간을 새로 업데이트할지 말지, 업데이트한다면 어떤 방향으로 업데이트할 지 판단할 때 사용
    phi_a = f(x_cur + a*p_cur)
    phi_b = f(x_cur + b*p_cur)
    dphi_a = grad_centraldiff(f, x_cur + a*p_cur)@p_cur
    
    dem = 2*(phi_b - phi_a - dphi_a*(b - a)) # 2차 함수 근사식 기반 minimum point 계산식의 분모
    
    if (abs(dem) < 1e-12) | (not np.isfinite(dem)): # 분모 수치적으로 불안정하면
        alpha_min = .5*(a + b) # 그냥 구간의 중심점을 minimum point로 상정하고 return
        return alpha_min
    else: # 분모 수치적으로 안정하면
        num = dphi_a*(b - a) # 분자 계산해주고
        alpha_min = a - num/dem # 2차근사식 minimum point 계산식 기반 minimum point 구해준다.
        alpha_min = np.clip(alpha_min, a + 0.1*(b - a), b - 0.1*(b - a)) # 만약 minimum point가 너무 작은 값이면(유의미한 point 이동 못 이뤄냄) 구간의 10% 지점으로, 너무 큰 값(너무 많이 point 이동해도 문제)이면 구간의 90% 지점으로 치환한다.
        return alpha_min    

# bracketing_alpha ⊂ wolfe_strong_interpol
def bracketing_alpha(f, x_cur, p_cur, c2_dphi0, phi_armijo, alpha_lo, alpha_hi): 
    # wolfe_strong_interpol에서 확정된 '큰' 골짜기 구간을 기반으로 alpha_optm 찾는 함수
    # 여기서 추가적으로 구간을 더 작게 업데이트할 수도 있음 -> '작은' 골짜기 구간
    phi_lo = f(x_cur+alpha_lo*p_cur)

    for _ in range(50):
        alpha_new = interpol_alpha(f, x_cur, p_cur, alpha_lo, alpha_hi) # 다항식 보간함수로 골짜기 구간에서의 최소추정점 alpha_new 구함
        phi_new = f(x_cur + alpha_new*p_cur) # alpha_new에서의 함수값
        dphi_new = grad_centraldiff(f, x_cur + alpha_new*p_cur)@p_cur # alpha_new에서의 기울기

        if (phi_new > phi_armijo(alpha_new)) | (phi_new >= phi_lo): # alpha_new에서의 함수값이 오히려 증가했다
            alpha_hi = alpha_new # alpha_optm은 alpha_lo와 alpha_new 사이 존재 -> '작은' 골짜기 구간 [alpha_lo, alpha_new]로 업뎃
            # alpha_lo unchanged
        else:
            if abs(dphi_new) <= c2_dphi0: # alpha_new에서의 함수값이 감소했고 기울기까지 작다
                alpha_optm = alpha_new # alpha_new가 alpha_optm
                return alpha_optm
            elif dphi_new > 0: # alpha_new에서의 함수값이 감소했는데 기울기는 여전히 양수다
                alpha_hi = alpha_new # alpha_optm은 alpha_lo와 alpha_new 사이 존재 -> '작은' 골짜기 구간 [alpha_lo, alpha_new]로 업뎃
                # alpha_lo unchanged
            else: # alpha_new에서의 함수값이 감소했는데 기울기가 음수다
                alpha_lo = alpha_new # alpha_optm은 alpha_new와 alpha_hi 사이 존재 -> '작은' 골짜기 구간 [alpha_new, alpha_hi]로 업뎃
                # alpha_hi unchanged
        
        phi_lo = f(x_cur + alpha_lo*p_cur) # 업뎃된 alpha_lo에서의 함수값 계산
        
        if abs(alpha_hi - alpha_lo) < 1e-8: # 만약 구간이 충분히 줄어들었으면 그냥 탈출해서 구간의 절반지점을 alpha_optm으로 return
            break
    
    return 0.5*(alpha_lo + alpha_hi)

# wolfe_strong_interpol - 확실한 '큰' 골짜기 구간을 찾아 거기서 alpha_optm을 찾는 함수
def wolfe_strong_interpol(f, x_cur, f_cur, grad_cur, p_cur, c2):
    c1 = 1e-4 # Armijo 조건, Curvature 조건용 factors
    alpha_try_old, alpha_try = 0, 1 # Initial bracket of alpha

    phi0 = f_cur # Armijo 함수 생성용
    dphi0 = grad_cur@p_cur # Armijo 함수 생성용 및 Curvature 조건 비교용

    phi_armijo = lambda alpha : phi0 + c1*alpha*dphi0 # Armijo 람다 함수 정의

    for _ in range(50):
        x_try = x_cur + alpha_try*p_cur
        phi_try = f(x_try)
        dphi_try = grad_centraldiff(f, x_try)@p_cur

        phi_armijo_try = phi_armijo(alpha_try)

        x_try_old = x_cur + alpha_try_old*p_cur
        phi_try_old = f(x_try_old)

        if (phi_try > phi_armijo_try) | (phi_try > phi_try_old): # phi_try가 충분히 크다면 -> alpha_optm이 alpha_try_old와 alpha_try 사이 존재
            alpha_lo, alpha_hi = alpha_try_old, alpha_try
            alpha_optm = bracketing_alpha(f, x_cur, p_cur, abs(c2*dphi0), phi_armijo, alpha_lo, alpha_hi) # bracketing 하고 interpolation iteration 돌려서 alpha_optm 뽑아내자
            return alpha_optm

        elif abs(dphi_try) <= -c2*dphi0: # phi_try가 충분히 작고 기울기까지 작다면
            alpha_optm = alpha_try # 그 점이 alph_optm이다
            return alpha_optm

        elif dphi_try >= 0: # phi_try가 충분히 작긴 한데 기울기가 양수라면 더 작은 phi 값을 가지는 alpha_optm이 alpha_try_old와 alpha_try 사이 존재
            alpha_lo, alpha_hi = alpha_try_old, alpha_try
            alpha_optm = bracketing_alpha(f, x_cur, p_cur, abs(c2*dphi0), phi_armijo, alpha_lo, alpha_hi) # bracketing 하고 interpolation iteration 돌려서 alpha_optm 뽑아내자
            return alpha_optm

        else: # phi_try가 충분히 작긴 한데 기울기가 음수라면 더 작은 phi 값을 가지는 alpha_optm은 alpha_try보다 뒤의 구간에 존재 -> 구간 업데이트
            alpha_try_old = alpha_try # 다음 구간의 하한 = 현재 구간의 상한
            alpha_try = min(alpha_try * 2, 10.0) # 다음 구간의 상한 = 현재 구간 상한의 2배. 최대는 10으로 제한
    
    if not np.isfinite(alpha_try):
        alpha_try = 1e-3
    return max(min(alpha_try, 1.0), 1e-6)

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
######################### Main Optimization Algorithm(Unconstrained) #########################
### Steepest Descent Method
# 가장 느리지만 가장 수렴 가능성 높음
def stp_descent(f, x0, tol):
    ### Check input data type
    if (not isinstance(x0, np.ndarray)) | (x0.ndim >= 2):
        raise ValueError('Please input 1D ndarray type !!')

    ### Check f(x0)
    f0 = f(x0)
    if not np.isfinite(f0).all():
        raise ValueError('Function value at x0 is not finite. Try another x0 !')

    ### Check ∇f(x0)
    grad0 = grad_centraldiff(f, x0)
    if np.linalg.norm(grad0) < tol: # Check optimality
        print(f'Since |grad(x0)| = {np.linalg.norm(grad0)} < {tol}, x0 : {x0} is optimum point !')
        list_x = [x0]; list_f = [f0]; list_grad = [grad0]
        return list_x, list_f, list_grad
    else:
        print(f'Since |grad(x0)| = {np.linalg.norm(grad0)} > {tol}, x0 : {x0} is not an optimum point. Optimization begins !')
        pass

    ### Initialization for searching iterations
    x_new = x0
    f_new = f0
    grad_new = grad0
    err = 1
    k = 0

    ############## 과제용 plot을 위한 log 담기 위한 list
    list_x = [x0]
    list_f = [f0]
    list_grad = [grad0]

    ############## Searching iterations
    ###### Update info of current point
    while err > tol:
        x_cur = x_new
        f_cur = f_new
        grad_cur = grad_new

        ###### Line search
        ### Search direction - by steepest gradient method
        p_cur = search_direction_stp_descent(f, x_cur)
        # Search direction check
        if grad_cur@p_cur > 0: # search direction이 증가 방향이면 경고 내보내고 steepest descent direction으로 search direction 변경
            print(f'Warning : grad(x_k)·p_k={grad_cur@p_cur} > 0 : Search direction p_k would likely make function increase !')
            print(f'Warning : p_k would be replaced with steepest descent direction grad(x_k) : {-grad_cur} !')
            p_cur = search_direction_stp_descent(f, x_cur)

        ### Step length - by Strong Wolfe + interpolation
        alpha_cur = wolfe_strong_interpol(f, x_cur, f_cur, grad_cur, p_cur, 0.9)

        ##### x_new update
        k = k + 1
        x_new = x_cur + alpha_cur*p_cur
        f_new = f(x_new)
        grad_new = grad_centraldiff(f, x_new)
        err = np.linalg.norm(grad_new)
        list_x.append(x_new)
        list_f.append(f_new)
        list_grad.append(grad_new)
        print(f'x_{k} : {x_new}')
        print(f'f_{k} : {f_new}')
        print(f'norm(grad(x_{k})) : {err}')
        print(f'recent alpha : {alpha_cur}')
        print(f'recent p : {p_cur}')
        print()

    print(f'Optimization converges -> Iteration : {k} / x* : {x_new} / f(x*) : {f(x_new)} / norm(grad(x*)) : {np.linalg.norm(grad_new)} ')
    return list_x, list_f, list_grad

### Congugate Gradient Method - Hestenes Stiefel(CGM-HS)
# 적당히 빠르고 수렴도 꽤 잘 됨
def cg_hs(f, x0, tol):
    ### Check input data type
    if (not isinstance(x0, np.ndarray)) | (x0.ndim >= 2):
        raise ValueError('Please input 1D ndarray type !!')

    ### Check f(x0)
    f0 = f(x0)
    if not np.isfinite(f0).all():
        raise ValueError('Function value at x0 is not finite. Try another x0 !')

    ### Check ∇f(x0)
    grad0 = grad_centraldiff(f, x0)
    if np.linalg.norm(grad0) < tol: # Check optimality
        print(f'Since |grad(x0)| = {np.linalg.norm(grad0)} < {tol}, x0 : {x0} is optimum point !')
        list_x = [x0]; list_f = [f0]; list_grad = [grad0]
        return list_x, list_f, list_grad
    else:
        print(f'Since |grad(x0)| = {np.linalg.norm(grad0)} > {tol}, x0 : {x0} is not an optimum point. Optimization begins !')
        pass

    ### Initialization for searching iterations
    p_cur = None
    grad_cur = None
    
    x_new = x0
    f_new = f0
    grad_new = grad0
    
    err = 1
    k = 0

    ############## 과제용 plot을 위한 log 담기 위한 list
    list_x = [x0]
    list_f = [f0]
    list_grad = [grad0]

    ############## Searching iterations
    ###### Update info of current point
    while err > tol:
        grad_old = grad_cur
        p_old = p_cur

        x_cur = x_new
        f_cur = f_new
        grad_cur = grad_new

        ###### Line search
        ### Search direction - by Conjugate Gradient_HS method
        p_cur = search_direction_cg_hs(k, grad_old, grad_cur, p_old)
        # Search direction check
        if grad_cur@p_cur > 0: # search direction이 증가 방향이면 경고 내보내고 steepest descent direction으로 search direction 변경
            print(f'Warning : grad(x_k)·p_k={grad_cur@p_cur} > 0 : Search direction p_k would likely make function increase !')
            print(f'Warning : p_k would be replaced with steepest descent direction grad(x_k) : {-grad_cur} !')
            p_cur = search_direction_stp_descent(f, x_cur)

        ### Step length - by Strong Wolfe + interpolation
        alpha_cur = wolfe_strong_interpol(f, x_cur, f_cur, grad_cur, p_cur, 0.1)

        ##### x_new update
        k = k + 1
        x_new = x_cur + alpha_cur*p_cur
        f_new = f(x_new)
        grad_new = grad_centraldiff(f, x_new)
        err = np.linalg.norm(grad_new)
        list_x.append(x_new)
        list_f.append(f_new)
        list_grad.append(grad_new)
        print(f'x_{k} : {x_new}')
        print(f'f_{k} : {f_new}')
        print(f'norm(grad(x_{k})) : {err}')
        print(f'recent alpha : {alpha_cur}')
        print(f'recent p : {p_cur}')
        print()

    print(f'Optimization converges -> Iteration : {k} / x* : {x_new} / f(x*) : {f(x_new)} / norm(grad(x*)) : {np.linalg.norm(grad_new)} ')
    return list_x, list_f, list_grad

### Congugate Gradient Method - Fletcher Reeves(CGM-FR)
# CGM-HS보다 안정적, 적당한 속도, 더 나은 수렴안정성
def cg_fr(f, x0, tol):
    ### Check input data type
    if (not isinstance(x0, np.ndarray)) | (x0.ndim >= 2):
        raise ValueError('Please input 1D ndarray type !!')

    ### Check f(x0)
    f0 = f(x0)
    if not np.isfinite(f0).all():
        raise ValueError('Function value at x0 is not finite. Try another x0 !')

    ### Check ∇f(x0)
    grad0 = grad_centraldiff(f, x0)
    if np.linalg.norm(grad0) < tol: # Check optimality
        print(f'Since |grad(x0)| = {np.linalg.norm(grad0)} < {tol}, x0 : {x0} is optimum point !')
        list_x = [x0]; list_f = [f0]; list_grad = [grad0]
        return list_x, list_f, list_grad
    else:
        print(f'Since |grad(x0)| = {np.linalg.norm(grad0)} > {tol}, x0 : {x0} is not an optimum point. Optimization begins !')
        pass

    ### Initialization for searching iterations
    p_cur = None
    grad_cur = None
    
    x_new = x0
    f_new = f0
    grad_new = grad0
    
    err = 1
    k = 0

    ############## 과제용 plot을 위한 log 담기 위한 list
    list_x = [x0]
    list_f = [f0]
    list_grad = [grad0]

    ############## Searching iterations
    ###### Update info of current point
    while err > tol:
        grad_old = grad_cur
        p_old = p_cur

        x_cur = x_new
        f_cur = f_new
        grad_cur = grad_new

        ###### Line search
        ### Search direction - by Conjugate Gradient_FR method
        p_cur = search_direction_cg_fr(k, grad_old, grad_cur, p_old)
        # Search direction check
        if grad_cur@p_cur > 0: # search direction이 증가 방향이면 경고 내보내고 steepest descent direction으로 search direction 변경
            print(f'Warning : grad(x_k)·p_k={grad_cur@p_cur} > 0 : Search direction p_k would likely make function increase !')
            print(f'Warning : p_k would be replaced with steepest descent direction grad(x_k) : {-grad_cur} !')
            p_cur = search_direction_stp_descent(f, x_cur)

        ### Step length - by Strong Wolfe + interpolation
        alpha_cur = wolfe_strong_interpol(f, x_cur, f_cur, grad_cur, p_cur, 0.1)

        ##### x_new update
        k = k + 1
        x_new = x_cur + alpha_cur*p_cur
        f_new = f(x_new)
        grad_new = grad_centraldiff(f, x_new)
        err = np.linalg.norm(grad_new)
        list_x.append(x_new)
        list_f.append(f_new)
        list_grad.append(grad_new)
        print(f'x_{k} : {x_new}')
        print(f'f_{k} : {f_new}')
        print(f'norm(grad(x_{k})) : {err}')
        print(f'recent alpha : {alpha_cur}')
        print(f'recent p : {p_cur}')
        print()

    print(f'Optimization converges -> Iteration : {k} / x* : {x_new} / f(x*) : {f(x_new)} / norm(grad(x*)) : {np.linalg.norm(grad_new)} ')
    return list_x, list_f, list_grad

### Newton's method
# 가장 빠른 수렴속도, 하지만 Hessian 계산량 비쌈, 또한 Hessian PD 아닐 경우 불안정 및 point에 따라 수렴안정성 낮아짐
def newton(f, x0, tol):
    ### Check input data type
    if (not isinstance(x0, np.ndarray)) | (x0.ndim >= 2):
        raise ValueError('Please input 1D ndarray type !!')

    ### Check f(x0)
    f0 = f(x0)
    if not np.isfinite(f0).all():
        raise ValueError('Function value at x0 is not finite. Try another x0 !')

    ### Check ∇f(x0)
    grad0 = grad_centraldiff(f, x0)
    if np.linalg.norm(grad0) < tol: # Check optimality
        print(f'Since |grad(x0)| = {np.linalg.norm(grad0)} < {tol}, x0 : {x0} is optimum point !')
        list_x = [x0]; list_f = [f0]; list_grad = [grad0]
        return list_x, list_f, list_grad
    else:
        print(f'Since |grad(x0)| = {np.linalg.norm(grad0)} > {tol}, x0 : {x0} is not an optimum point. Optimization begins !')
        pass

    ### Initialization for searching iterations
    p_cur = None
    grad_cur = None
    
    x_new = x0
    f_new = f0
    grad_new = grad0
    
    err = 1
    k = 0

    ############## 과제용 plot을 위한 log 담기 위한 list
    list_x = [x0]
    list_f = [f0]
    list_grad = [grad0]    

    ############## Searching iterations
    ###### Update info of current point
    while err > tol:
        x_cur = x_new
        f_cur = f_new
        grad_cur = grad_new
        hessian_cur = hessian_centraldiff(f, x_cur)

        ###### Line search
        ### Search direction - by Newton method
        p_cur = search_direction_newton(grad_cur, hessian_cur)

        # Search direction check
        if grad_cur@p_cur > 0: # search direction이 증가 방향이면 경고 내보내고 steepest descent direction으로 search direction 변경
            print(f'Warning : grad(x_k)·p_k={grad_cur@p_cur} > 0 : Search direction p_k would likely make function increase !')
            print(f'Warning : p_k would be replaced with steepest descent direction grad(x_k) : {-grad_cur} !')
            p_cur = search_direction_stp_descent(f, x_cur)

        ### Step length - by Strong Wolfe + interpolation
        alpha_cur = wolfe_strong_interpol(f, x_cur, f_cur, grad_cur, p_cur, 0.9)

        ##### x_new update
        k = k + 1
        x_new = x_cur + alpha_cur*p_cur
        f_new = f(x_new)
        grad_new = grad_centraldiff(f, x_new)
        err = np.linalg.norm(grad_new)
        list_x.append(x_new)
        list_f.append(f_new)
        list_grad.append(grad_new)
        print(f'x_{k} : {x_new}')
        print(f'f_{k} : {f_new}')
        print(f'norm(grad(x_{k})) : {err}')
        print(f'recent alpha : {alpha_cur}')
        print(f'recent p : {p_cur}')
        print()

    print(f'Optimization converges -> Iteration : {k} / x* : {x_new} / f(x*) : {f(x_new)} / norm(grad(x*)) : {np.linalg.norm(grad_new)} ')
    return list_x, list_f, list_grad

### Quasi-Newton's Method - BFGS
# Newton's method와 유사한 속도, 낮은 근사 Hessian 계산량, Hessain의 PD를 보장하여 수렴안정성 비교적 높음
def quasi_newton_bfgs(f, x0, tol):
    ### Check input data type
    if (not isinstance(x0, np.ndarray)) | (x0.ndim >= 2):
        raise ValueError('Please input 1D ndarray type !!')

    ### Check f(x0)
    f0 = f(x0)
    if not np.isfinite(f0).all():
        raise ValueError('Function value at x0 is not finite. Try another x0 !')

    ### Check ∇f(x0)
    grad0 = grad_centraldiff(f, x0)
    if np.linalg.norm(grad0) < tol: # Check optimality
        print(f'Since |grad(x0)| = {np.linalg.norm(grad0)} < {tol}, x0 : {x0} is optimum point !')
        list_x = [x0]; list_f = [f0]; list_grad = [grad0]
        return list_x, list_f, list_grad
    else:
        print(f'Since |grad(x0)| = {np.linalg.norm(grad0)} > {tol}, x0 : {x0} is not an optimum point. Optimization begins !')
        pass

    ### Initialization for searching iterations
    x_cur = None
    grad_cur = None
    hessian_inv_aprx_cur = np.identity(len(x0))

    x_new = x0
    f_new = f0
    grad_new = grad0
    
    err = 100
    k = 0

    ############## 과제용 plot을 위한 log 담기 위한 list
    list_x = [x0]
    list_f = [f0]
    list_grad = [grad0]

    ############## Searching iterations
    ###### Update info of current point
    while err > tol:
        x_old = x_cur
        grad_old = grad_cur
        hessian_inv_aprx_old = hessian_inv_aprx_cur
        
        x_cur = x_new
        f_cur = f_new
        grad_cur = grad_new

        ###### Line search
        ### Search direction - by Quasi-Newton's(BFGS) method
        p_cur, hessian_inv_aprx_cur = search_direction_quasi_newton_bfgs(k, x_old, x_cur, grad_old, grad_cur, hessian_inv_aprx_old)

        # Search direction check
        if grad_cur@p_cur > 0: # search direction이 증가 방향이면 경고 내보내고 steepest descent direction으로 search direction 변경
            print(f'Warning : grad(x_k)·p_k={grad_cur@p_cur} > 0 : Search direction p_k would likely make function increase !')
            print(f'Warning : p_k would be replaced with steepest descent direction grad(x_k) : {-grad_cur} !')
            p_cur = search_direction_stp_descent(f, x_cur)

        ### Step length - by Strong Wolfe + interpolation
        alpha_cur = wolfe_strong_interpol(f, x_cur, f_cur, grad_cur, p_cur, 0.9)

        ##### x_new update
        k = k + 1
        x_new = x_cur + alpha_cur*p_cur
        f_new = f(x_new)
        grad_new = grad_centraldiff(f, x_new)
        err = np.linalg.norm(grad_new)
        list_x.append(x_new)
        list_f.append(f_new)
        list_grad.append(grad_new)
        print(f'x_{k} : {x_new}')
        print(f'f_{k} : {f_new}')
        print(f'norm(grad(x_{k})) : {err}')
        print(f'recent alpha : {alpha_cur}')
        print(f'recent p : {p_cur}')
        print()

    print(f'Optimization converges -> Iteration : {k} / x* : {x_new} / f(x*) : {f(x_new)} / norm(grad(x*)) : {np.linalg.norm(grad_new)} ')
    return list_x, list_f, list_grad

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
########################## Main Optimization Algorithm(Constrained) ##########################
### Quadratic Penalty Method(QPM)
# Heuristic method로서 penalty parameter(mu)를 키워가며 constraint를 잡는 전략
# penalty parameter(mu)가 너무 커질 경우 inner loop subproblem이 ill-conditioned하게 되어 수렴 불안정.
def qpm(f, ce, ci, x0, inner_opt, tol):
    ### Check input data type
    if (not isinstance(ce, list)) | (not isinstance(ci, list)) | (len(ci) + len(ce) == 0):
        raise ValueError('Please input at least either one equality or inequality constraint as list type ! ; Empty list is OK as well.')

    if (not isinstance(x0, np.ndarray)) | (x0.ndim >= 2):
        raise ValueError('Please input x0 as 1D ndarray type !!')
    
    ### Check inner loop optimizer
    if inner_opt == 0:
        inner_opt = stp_descent
    elif inner_opt == 1:
        inner_opt = cg_hs
    elif inner_opt == 2:
        inner_opt = cg_fr
    elif inner_opt == 3:
        inner_opt = quasi_newton_bfgs
    else:
        raise ValueError('Please input correct integer for inner_opt ! ; 0:stp_descent, 1:cg_hs, 2:cg_fr, 3:quasi_newton_bfgs')

    ### Check f(x0)
    f0 = f(x0)
    if not np.isfinite(f0).all():
        raise ValueError('Function value at x0 is not finite. Try another x0 !')

    ### Check ci(x0) ≥ 0
    if len(ci) >= 1: # 부등호제약조건은 feasibility 고려하고 등호제약조건은 미고려(∵ exactly하게 맞추기가 더 어려움)
        infeasible_ci = [ci_i(x0) for ci_i in ci if ci_i(x0) < 0] # infeasible criteria of c
        if len(infeasible_ci) >= 1:
            raise ValueError(f'Infeasible x0 for {len(infeasible_ci)} of {len(ci)} inequality constraint(s). Try feasible x0 !')

    ### Check ∇f(x0)
    grad0 = grad_centraldiff(f, x0)
    if np.linalg.norm(grad0) < tol: # Check optimality
        print(f'Since |grad(x0)| = {np.linalg.norm(grad0)} < {tol}, x0 : {x0} is optimum point !')
        # return x0

    ### constraints manipulation
    if len(ce) == 0:
        ce = [lambda x : 0]
    if len(ci) == 0:
        ci = [lambda x : 0]

    ### Initialization for searching iterations
    x_new = x0
    # Parameter setting for QPM
    # Qk 만들 때 필요한 quadratic penalty term of constraints 미리 정의
    sum_ce_sq = lambda x : np.sum([(ce_i(x))**2 for ce_i in ce]) # sum(c_e(x)^2) for Qk
    sum_ci_sq = lambda x : np.sum([(np.max([-ci_i(x), 0]))**2 for ci_i in ci]) # sum(c_i(x)^2) for Qk

    f_mu = 5; mu_new = 1 # increase factor for penalty parameter mu ; mu0 = 1
    f_tau = .5; tau_new = .2 # decrease factor for convergence criteria for Qk ; tau0 = .2

    ### 과제용 plot을 위한 log 담기 위한 list
    list_x = [x0]
    list_f = [f0]
    list_ce = [[ce_i(x0) for ce_i in ce]]
    list_ci = [[ci_i(x0) for ci_i in ci]]

    ### Outer loop begins
    for j in np.arange(100): # mu가 너무 커지면 Qk가 unstable해지기 때문에 어차피 finite한 iterations 내에서 쇼부를 봐야 한다.
        # Finding a x*_k in Qk
        x_cur = x_new # x_k
        mu_cur = mu_new; print(f'mu_{j} = {mu_cur}') # penalty parameter update(increase)
        tau_cur = tau_new; print(f'tau_{j} = {tau_cur}') # Qk convergence criteria update(decrease)
        Q_cur = lambda x : f(x) + .5*mu_cur*sum_ce_sq(x) + .5*mu_cur*sum_ci_sq(x) # quadratic penalty function Qk at x_k
        log_inner = inner_opt(Q_cur, x_cur, tau_cur) # Solving ∇Qk(x*_k) ≤ tau_k ; Inner loop
        x_new = log_inner[0][-1]; f_new = f(x_new); ce_new = [ce_i(x_new) for ce_i in ce]; ci_new = [ci_i(x_new) for ci_i in ci]
        mu_new = mu_cur*f_mu
        tau_new = tau_cur*f_tau

        # Convergence check of x* ... QPM을 위한 convergence 기준은 명확한 게 없음. 애초에 휴리스틱 알고리즘이라 근본이 없는 놈이라 그럼.
        move_x = np.linalg.norm(x_new - x_cur) # convergence criteria of x*_k
        violated_ce = [ce_new_i for ce_new_i in ce_new if np.abs(ce_new_i) > 1e-4] # violation criteria of ce(x*_k)
        violated_ci = [ci_new_i for ci_new_i in ci_new if ci_new_i < -1e-4] # violation criteria of ci(x*_k)
        if (move_x < 1e-4) & (len(violated_ci) == 0) & (len(violated_ce) == 0):
            done = True # flag for termination of outerloop
        else:
            done = False

        print(f'{j+1}-th outer loop : Inner loop converges at {len(log_inner[0]) - 1} iteration(s) ...')
        print(f'|x_{j+1} - x_{j}| = {move_x}')
        print(f'# of violated ce constraints : {len(violated_ce)}, violation : {np.sum(violated_ce)}')
        print(f'# of violated ci constraints : {len(violated_ci)}, violation : {np.sum(violated_ci)}')
        print(f'\n------------------------------------------------------------- Outer loop ----------------------------------------------------------------\n')

        list_x.append(x_new)
        list_f.append(f_new)
        list_ce.append(ce_new)
        list_ci.append(ci_new)

        if done:
            print(f'Outer loop converged at {j+1} iteration(s) !')
            print(f'iter = {j+1} x* = {x_new}, f(x*) = {f(x_new)}, |ce(x*)|₁ = {np.sum(np.abs(ce_new))}, |ci(x*)|₁ = {np.sum(np.abs(ci_new))}')
            return list_x, list_f, list_ce, list_ci

### Augmented Lagrangian Method(ALM)
# KKT conditions based on Lagrangian function 개념을 활용하여 비교적 작은 penalty method(mu, rho)로도 안정적 수렴 가능.

def alm(f, ce, ci, x0, inner_opt, tol):
    ### Check input data type
    if (not isinstance(ce, list)) | (not isinstance(ci, list)) | (len(ci) + len(ce) == 0):
        raise ValueError('Please input at least either one equality or inequality constraint as list type ! ; Empty list is OK as well.')

    if (not isinstance(x0, np.ndarray)) | (x0.ndim >= 2):
        raise ValueError('Please input x0 as 1D ndarray type !!')

    ### Check inner loop optimizer
    if inner_opt == 0:
        inner_opt = stp_descent
    elif inner_opt == 1:
        inner_opt = cg_hs
    elif inner_opt == 2:
        inner_opt = cg_fr
    elif inner_opt == 3:
        inner_opt = quasi_newton_bfgs
    else:
        raise ValueError('Please input correct integer for inner_opt ! ; 0:stp_descent, 1:cg_hs, 2:cg_fr, 3:quasi_newton_bfgs')

    ### Check f(x0)
    f0 = f(x0)
    if not np.isfinite(f0).all():
        raise ValueError('Function value at x0 is not finite. Try another x0 !')

    ### Check ci(x0) ≥ 0
    if len(ci) >= 1: # 부등호제약조건은 feasibility 고려하고 등호제약조건은 미고려(∵ exactly하게 맞추기가 더 어려움)
        infeasible_ci = [ci_j(x0) for ci_j in ci if ci_j(x0) < 0] # infeasible criteria of c
        if len(infeasible_ci) >= 1:
            raise ValueError(f'Infeasible x0 for {len(infeasible_ci)} of {len(ci)} inequality constraint(s). Try feasible x0 !')

    ### Check ∇f(x0)
    grad0 = grad_centraldiff(f, x0)
    if np.linalg.norm(grad0) < tol: # Check optimality
        print(f'Since |grad(x0)| = {np.linalg.norm(grad0)} < {tol}, x0 : {x0} is optimum point !')
        # return x0

    ### constraints manipulation
    if len(ce) == 0:
        ce = [lambda x : 0]
    if len(ci) == 0:
        ci = [lambda x : 0]

    ### Initialization for searching iterations
    x_new = x0

    ### Parameter setting for ALM
    # Final tolerance for outer loop
    tol_eq_final = 1e-6 # equality feasibility
    tol_ineq_final = 1e-6 # inequality feasibility
    tol_opt_final = 1e-6 # optimality (∥∇L_A∥∞)
    tol_step_final = 1e-8 # step size (relative factor 포함 권장)

    # Initial tolearnce of constraints for outer loop(점차 tighten)
    tol_eq   = 1e-3
    tol_ineq = 1e-3

    # Tolerance update scheme for inner loop : tau_k = max(omega_min, omega0 * (omega_decay)^k)
    omega0 = 1e-2
    omega_decay = 0.5
    omega_min = tol_opt_final
    tau = omega0   # == tau_0

    # Penalty parameter increase factor / upper bound
    factor_mu  = 5.0
    factor_rho = 5.0
    mu_max   = 1e8
    rho_max  = 1e8

    lmbda = np.array([0]*len(ce)) # initial lagrange multipliers of equality constraints for Lk
    nu = np.array([0]*len(ci)) # initial lagrange multipliers of inequality constraints
    mu = 1 # increase factor for equality constraint penalty parameter mu ; mu0 = 1
    rho = 1 # increase factor for inequality constraint penalty parameter rho ; rho0 = 1

    ### 과제용 plot을 위한 log 담기 위한 list
    list_x = [x0]
    list_f = [f0]
    list_grad = [grad0]
    list_ce = [[ce_i(x0) for ce_i in ce]]
    list_ci = [[ci_i(x0) for ci_i in ci]]
    list_lmbda = [lmbda]
    list_nu = [nu]

    ### Outer loop begins
    for k in np.arange(100): # mu가 너무 커지면 Qk가 unstable해지기 때문에 어차피 finite한 iterations 내에서 쇼부를 봐야 한다.
        x_cur = x_new # x_k
        print(f'mu_{k} = {mu}')
        print(f'rho_{k} = {rho}')
        print(f'tau_{k} = {tau}')

        # penalty term update
        penalty_ce = lambda x : -lmbda@np.array([ce_j(x) for ce_j in ce]) + .5*mu*np.sum(np.array([ce_j(x)**2 for ce_j in ce]))
        penalty_ci = lambda x : (-nu@np.array([ci_j(x) - np.maximum(ci_j(x) - nu[j]/rho, 0) for j, ci_j in enumerate(ci)]) +
                                    .5*rho*np.sum(np.array([(ci_j(x) - np.maximum(ci_j(x) - nu[j]/rho, 0))**2 for j, ci_j in enumerate(ci)])))

        LA_cur = lambda x : f(x) + penalty_ce(x) + penalty_ci(x) # augmented lagrangian function LAk at x_k
        log_inner = inner_opt(LA_cur, x_cur, tau) # solving ∇LAk(x*_k) ≤ tau_k ; Inner loop
        x_new = log_inner[0][-1]
        f_new = f(x_new)
        ce_new = np.array([ce_j(x_new) for ce_j in ce])
        ci_new = np.array([ci_j(x_new) for ci_j in ci])

        # --- multiplier updates (AFTER computing ce_new, ci_new) ---
        if len(ce) >= 1:
            lmbda = lmbda - mu * ce_new
        if len(ci) >= 1:
            nu = np.maximum(nu - rho * ci_new, 0.0)

        grad_LA_new = log_inner[-1][-1]

        # residual(잔차) 계산
        r_ce = np.max(np.abs(ce_new)) if len(ce_new) >= 1 else 0 # 등호제약조건 잔차(위반)
        r_ci = np.max(np.maximum(-ci_new, 0)) if len(ci_new) >= 1 else 0 # 부등호제약조건 잔차(위반)
        r_grad_LA = np.linalg.norm(grad_LA_new, ord=np.inf) # ∇L_A 수준
        r_step = np.linalg.norm(x_new - x_cur) # x_new - x_cur 거리

        # Convergence check for Outer loop
        if ((r_ce <= tol_eq_final) & # 등호제약조건 위반이 충분히 작고
            (r_ci <= tol_ineq_final) & # 부등호제약조건 위반도 충분히 작고
            (r_grad_LA <= tol_opt_final) & # ∇L_A도 충분히 정칙점에 도달했고
            (r_step <= tol_step_final * (1.0 + np.linalg.norm(x_new)))): # x_new도 충분히 수렴했다면
            done = True # outer loop 종료 flag 마킹하자
        else:
            done = False

        # ---- penalty parameter update(keep or increase) ----
        # ce
        if r_ce <= tol_eq:
            mu = mu # 등호제약조건 위반이 충분히 작다면 페널티 파라미터를 그대로 두자
        else:
            mu = min(factor_mu*mu, mu_max) # 등호제약조건 위반이 크다면 페널티 파라미터를 증가시키자

        # ci
        if r_ci <= tol_ineq:
            rho = rho # 부등호제약조건 위반이 충분히 작다면 페널티 파라미터를 그대로 두자
        else:
            rho = min(factor_rho*rho, rho_max) # 부등호제약조건 위반이 크다면 페널티 파라미터를 증가시키자

        # ---- tolerance update ----
        # If 제약조건 잔차 ≈ 0 and ∇L_A ≈ 0 -> 제약조건 tolerance를 조금 더 빡세게 두자(감소시키자)
        if (r_ce <= 0.3*tol_eq) & (r_ci <= 0.3*tol_ineq) & (r_grad_LA <= 0.3*tau):
            tol_eq = max(tol_eq_final,   0.5*tol_eq)
            tol_ineq = max(tol_ineq_final, 0.5*tol_ineq)

        # 내부 허용치 스케줄(항상 단조 감소)
        tau = max(omega_min, omega_decay*tau)

        # ---------------------------------------- print log ----------------------------------------
        print(f'{k+1}-th outer loop : Inner loop converges at {len(log_inner[0]) - 1} iteration(s) ...')
        print(f'|x_{k+1} - x_{k}| = {r_step}')
        print(f'Max violation of equality constraints : {r_ce}')
        print(f'Max violation of inequality constraints : {r_ci}')
        print(f'\n------------------------------------------------------------- Outer loop ----------------------------------------------------------------\n')

        list_x.append(x_new)
        list_f.append(f_new)
        list_grad.append(grad_LA_new)
        list_ce.append(ce_new)
        list_ci.append(ci_new)
        list_lmbda.append(lmbda)
        list_nu.append(nu)

        if done:
            print(f'Outer loop converges at {k+1} iteration(s) !')
            print(f'iter = {k+1} x* = {x_new}, f(x*) = {f(x_new)}, ∇L_A(x*) = {grad_LA_new}, max(ce(x*)) = {r_ce}, max(ci(x*)) = {r_ci}')
            return list_x, list_f, list_grad, list_ce, list_ci, list_lmbda, list_nu
        
    print(f'Outer loop terminates at {k+1}(max) iteration(s) !')
    print(f'iter = {k+1} x* = {x_new}, f(x*) = {f(x_new)}, ∇L_A(x*) = {grad_LA_new}, max(ce(x*)) = {r_ce}, max(ci(x*)) = {r_ci}')
    return list_x, list_f, list_grad, list_ce, list_ci, list_lmbda, list_nu

def alm4sqp(f, ce, ci, x0, lmbda0, nu0, inner_opt, tol):
    ### Check input data type
    if (not isinstance(ce, list)) | (not isinstance(ci, list)) | (len(ci) + len(ce) == 0):
        raise ValueError('Please input at least either one equality or inequality constraint as list type ! ; Empty list is OK as well.')

    if (not isinstance(x0, np.ndarray)) | (x0.ndim >= 2):
        raise ValueError('Please input x0 as 1D ndarray type !!')

    ### Check inner loop optimizer
    if inner_opt == 0:
        inner_opt = stp_descent
    elif inner_opt == 1:
        inner_opt = cg_hs
    elif inner_opt == 2:
        inner_opt = cg_fr
    elif inner_opt == 3:
        inner_opt = quasi_newton_bfgs
    else:
        raise ValueError('Please input correct integer for inner_opt ! ; 0:stp_descent, 1:cg_hs, 2:cg_fr, 3:quasi_newton_bfgs')

    ### Check f(x0)
    f0 = f(x0)
    if not np.isfinite(f0).all():
        raise ValueError('Function value at x0 is not finite. Try another x0 !')

    # ### Check ci(x0) ≥ 0
    # if len(ci) >= 1: # 부등호제약조건은 feasibility 고려하고 등호제약조건은 미고려(∵ exactly하게 맞추기가 더 어려움)
    #     infeasible_ci = [ci_j(x0) for ci_j in ci if ci_j(x0) < 0] # infeasible criteria of c
    #     if len(infeasible_ci) >= 1:
    #         raise ValueError(f'Infeasible x0 for {len(infeasible_ci)} of {len(ci)} inequality constraint(s). Try feasible x0 !')

    ### Check ∇f(x0)
    grad0 = grad_centraldiff(f, x0)
    if np.linalg.norm(grad0) < tol: # Check optimality
        # print(f'Since |grad(x0)| = {np.linalg.norm(grad0)} < {tol}, x0 : {x0} is optimum point !')
        # return x0
        pass
    
    ### constraints manipulation
    if len(ce) == 0:
        ce = [lambda x : 0]
    if len(ci) == 0:
        ci = [lambda x : 0]

    ### Initialization for searching iterations
    x_new = x0

    ### Parameter setting for ALM
    # Final tolerance for outer loop
    tol_eq_final = 1e-6 # equality feasibility
    tol_ineq_final = 1e-6 # inequality feasibility
    tol_opt_final = 1e-6 # optimality (∥∇L_A∥∞)
    tol_step_final = 1e-8 # step size (relative factor 포함 권장)

    # Initial tolearnce of constraints for outer loop(점차 tighten)
    tol_eq   = 1e-3
    tol_ineq = 1e-3

    # Tolerance update scheme for inner loop : tau_k = max(omega_min, omega0 * (omega_decay)^k)
    omega0 = 1e-2
    omega_decay = 0.5
    omega_min = tol_opt_final
    tau = omega0   # == tau_0

    # Penalty parameter increase factor / upper bound
    factor_mu  = 5.0
    factor_rho = 5.0
    mu_max   = 1e8
    rho_max  = 1e8

    lmbda = lmbda0 # initial lagrange multipliers of equality constraints for Lk
    nu = nu0 # initial lagrange multipliers of inequality constraints
    mu = 1 # increase factor for equality constraint penalty parameter mu ; mu0 = 1
    rho = 1 # increase factor for inequality constraint penalty parameter rho ; rho0 = 1

    ### 과제용 plot을 위한 log 담기 위한 list
    list_x = [x0]
    list_f = [f0]
    list_grad = [grad0]
    list_ce = [[ce_i(x0) for ce_i in ce]]
    list_ci = [[ci_i(x0) for ci_i in ci]]
    list_lmbda = [lmbda]
    list_nu = [nu]

    ### Outer loop begins
    for k in np.arange(100): # mu가 너무 커지면 Qk가 unstable해지기 때문에 어차피 finite한 iterations 내에서 쇼부를 봐야 한다.
        x_cur = x_new # x_k
        # print(f'mu_{k} = {mu}')
        # print(f'rho_{k} = {rho}')
        # print(f'tau_{k} = {tau}')

        # penalty term update
        penalty_ce = lambda x : -lmbda@np.array([ce_j(x) for ce_j in ce]) + .5*mu*np.sum(np.array([ce_j(x)**2 for ce_j in ce]))
        penalty_ci = lambda x : (-nu@np.array([ci_j(x) - np.maximum(ci_j(x) - nu[j]/rho, 0) for j, ci_j in enumerate(ci)]) +
                                    .5*rho*np.sum(np.array([(ci_j(x) - np.maximum(ci_j(x) - nu[j]/rho, 0))**2 for j, ci_j in enumerate(ci)])))

        LA_cur = lambda x : f(x) + penalty_ce(x) + penalty_ci(x) # augmented lagrangian function LAk at x_k
        with io.StringIO() as buf, contextlib.redirect_stdout(buf):
            log_inner = inner_opt(LA_cur, x_cur, tau) # solving ∇LAk(x*_k) ≤ tau_k ; Inner loop
        x_new = log_inner[0][-1]
        f_new = f(x_new)
        ce_new = np.array([ce_j(x_new) for ce_j in ce])
        ci_new = np.array([ci_j(x_new) for ci_j in ci])

        # --- multiplier updates (AFTER computing ce_new, ci_new) ---
        if len(ce) >= 1:
            lmbda = lmbda - mu * ce_new
        if len(ci) >= 1:
            nu = np.maximum(nu - rho * ci_new, 0.0)

        grad_LA_new = log_inner[-1][-1]

        # residual(잔차) 계산
        r_ce = np.max(np.abs(ce_new)) if len(ce_new) >= 1 else 0 # 등호제약조건 잔차(위반)
        r_ci = np.max(np.maximum(-ci_new, 0)) if len(ci_new) >= 1 else 0 # 부등호제약조건 잔차(위반)
        r_grad_LA = np.linalg.norm(grad_LA_new, ord=np.inf) # ∇L_A 수준
        r_step = np.linalg.norm(x_new - x_cur) # x_new - x_cur 거리

        # Convergence check for Outer loop
        if ((r_ce <= tol_eq_final) & # 등호제약조건 위반이 충분히 작고
            (r_ci <= tol_ineq_final) & # 부등호제약조건 위반도 충분히 작고
            (r_grad_LA <= tol_opt_final) & # ∇L_A도 충분히 정칙점에 도달했고
            (r_step <= tol_step_final * (1.0 + np.linalg.norm(x_new)))): # x_new도 충분히 수렴했다면
            done = True # outer loop 종료 flag 마킹하자
        else:
            done = False

        # ---- penalty parameter update(keep or increase) ----
        # ce
        if r_ce <= tol_eq:
            mu = mu # 등호제약조건 위반이 충분히 작다면 페널티 파라미터를 그대로 두자
        else:
            mu = min(factor_mu*mu, mu_max) # 등호제약조건 위반이 크다면 페널티 파라미터를 증가시키자

        # ci
        if r_ci <= tol_ineq:
            rho = rho # 부등호제약조건 위반이 충분히 작다면 페널티 파라미터를 그대로 두자
        else:
            rho = min(factor_rho*rho, rho_max) # 부등호제약조건 위반이 크다면 페널티 파라미터를 증가시키자

        # ---- tolerance update ----
        # If 제약조건 잔차 ≈ 0 and ∇L_A ≈ 0 -> 제약조건 tolerance를 조금 더 빡세게 두자(감소시키자)
        if (r_ce <= 0.3*tol_eq) & (r_ci <= 0.3*tol_ineq) & (r_grad_LA <= 0.3*tau):
            tol_eq = max(tol_eq_final,   0.5*tol_eq)
            tol_ineq = max(tol_ineq_final, 0.5*tol_ineq)

        # 내부 허용치 스케줄(항상 단조 감소)
        tau = max(omega_min, omega_decay*tau)

        # # ---------------------------------------- print log ----------------------------------------
        # print(f'{k+1}-th outer loop : Inner loop converges at {len(log_inner[0]) - 1} iteration(s) ...')
        # print(f'|x_{k+1} - x_{k}| = {r_step}')
        # print(f'Max violation of equality constraints : {r_ce}')
        # print(f'Max violation of inequality constraints : {r_ci}')
        # print(f'\n------------------------------------------------------------- Outer loop ----------------------------------------------------------------\n')

        list_x.append(x_new)
        list_f.append(f_new)
        list_grad.append(grad_LA_new)
        list_ce.append(ce_new)
        list_ci.append(ci_new)
        list_lmbda.append(lmbda)
        list_nu.append(nu)

        if done:
            # print(f'Outer loop converges at {k+1} iteration(s) !')
            # print(f'iter = {k+1} x* = {x_new}, f(x*) = {f(x_new)}, ∇L_A(x*) = {grad_LA_new}, max(ce(x*)) = {r_ce}, max(ci(x*)) = {r_ci}')
            return list_x, list_f, list_grad, list_ce, list_ci, list_lmbda, list_nu
        
    # print(f'Outer loop terminates at {k+1}(max) iteration(s) !')
    # print(f'iter = {k+1} x* = {x_new}, f(x*) = {f(x_new)}, ∇L_A(x*) = {grad_LA_new}, max(ce(x*)) = {r_ce}, max(ci(x*)) = {r_ci}')
    return list_x, list_f, list_grad, list_ce, list_ci, list_lmbda, list_nu