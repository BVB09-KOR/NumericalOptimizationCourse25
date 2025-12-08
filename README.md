# For Numerical Optimization Course(MECH 579) 2025 Fall Semester, Mech Eng, McGill  
## Line Search and Gradient Based Multivariate Optimization(Unconstrained & Constrained)  

Each unconstrained/constrained optimizer is used as a function from `module_opt.py`(or `module_opt_AD.py`)  
Users simply define a single scalar objective function and list of equality/inequality constraint functions and provide them as input arguments to desired optimizer with proper initial guess point `x0`(convergence tolerance is supplementary).  
 - `module_opt.py ` 
   → Module containing Optimization Functions using Finite Difference based Gradient Calculation  
 - `module_opt_AD.py`
   → Module containing Optimization Functions using Automatic Differentiation(provided by PyTorch) based Gradient Calculation  
   → You should provide two types of functions as input argument, one with only numpy operations and another with only torch operations(treat only torch.Tensor datatype as input and output) when using each optimizer  

---------------------------------------------------------------------------------------------------

### Unconstrained Optimization Solver  
1. Steepest Descent Method(SDM) : `stp_descent(f, x0, tol)`  
   1) The output of SDM contains log of $x_k, f\(x_k\), ∇f\(x_k\)$  
3. Conjugate Gradient Method(CGM) : `cg_hs(f, x0, tol)` / `cg_fr(f, x0, tol)`  
   1) The output of CGM contains log of $x_k, f\(x_k\), ∇f\(x_k\)$  
5. Newton's Method(only in module_opt.py) : `newton(f, x0, tol)`  
   1) The output of Newton's Method contains log of $x_k, f\(x_k\), ∇f\(x_k\)$  
7. Quasi Newton's Method(QNM) - BFGS : `quasi_newton_bfgs(f, x0, tol)`  
   1) The output of QNM contains log of $x_k, f\(x_k\), ∇f\(x_k\)$  

 - Test results.  
   - module_opt_unconstrained_test.ipynb  
   - module_opt_unconstrained_AD_test.ipynb  

---------------------------------------------------------------------------------------------------

### Constrained Optimization Solver  
1. Quadratic Penalty Method(QPM) : `qpm(f, ce, ci, x0, inner_opt, tol)`  
   1) The output of QPM contains log of $x_k, f\(x_k\), ∇f\(x_k\), max\(|c_e\(x_k\)|\), max\(|c_i\(x_k\)|\)$  
3. Augmented Lagrangian Method(ALM) : `alm(f, ce, ci, x0, inner_opt, tol)`  
   1) ALM builds up unconstrained opt problem based on augmented Lagrangian function of original constrained opt problem at each k-th iteration  
   2) Each unconstrained opt problem is solved using unconstrained optimizer  
   3) The output of ALM contains log of $x_k, f\(x_k\), ∇L\(x_k\), max\(|c_e\(x_k\)|\), max\(|c_i\(x_k\)|\), \\lambda_k, \\nu_k$  
4. Sequential Quadratic Programming(with ALM) : `sqp(f, ce, ci, x0, maxiter=100, inner_opt=3, tol=1e-6, tol_inter=1e-4)`  
   1) SQP loop builds up Quadratic Programming subproblem(QPk) at each k-th iteration  
   2) QPk is solved using `alm4sqp(f, ce, ci, x0, lmbda0, nu0, inner_opt, tol)`  
      - `alm4sqp()` receives $p_{k-1}^*, \\lambda_k, \\nu_k$ from SQP loop as initial guess of design variables(`x0`), and lagrange multipliers `lmbda0`, `nu0`.  
   4) The output of SQP contains log of $x_k, f\(x_k\), ∇f\(x_k\), ∇L\(x_k\), max\(|c_e\(x_k\)|\), max\(|c_i\(x_k\)|\), \\lambda_k, \\nu_k$  
  
 - Test results.  
   - sqp_test_important.ipynb
   - assignment3.ipynb
   - assignment3_AD.ipynb
   - assignment5_*.ipynb ... refer to the assignment_5 pdf file.  

