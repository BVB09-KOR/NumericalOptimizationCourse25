## For Numerical Optimization Course(MECH 579) 2025 Fall Semester, Mech Eng, McGill  
### Line Search and Gradient Based Multivariate Optimization(Unconstrained & Constrained)  

Each unconstrained/constrained optimizer is used as a function from module_opt.py(or module_opt_AD.py)  
Users simply define a single scalar objective function and list of equality/inequality constraint functions and provide them as input arguments to desired optimizer with proper initial guess point x0(convergence tolerance is supplementary).  
 - module_opt.py ... Module containing Optimization Functions using Finite Difference based Gradient Calculation  
 - module_opt_AD.py ... Module containing Optimization Functions using Automatic Differentiation(provided by PyTorch) based Gradient Calculation
   - You should provide two types of functions as input argument, one with only numpy operations and another with only torch operations(treat only torch.Tensor datatype as input and output) when using each optimizer  

---------------------------------------------------------------------------------------------------

Unconstrained Optimization Solver  
1. Steepest Descent Method : stp_descent(f, x0, tol)  
2. Conjugate Gradient Method : cg_hs(f, x0, tol) / cg_fr(f, x0, tol)  
3. Newton's Method(only in module_opt.py) : newton(f, x0, tol)  
4. Quasi Newton's Method - BFGS : quasi_newton_bfgs(f, x0, tol)  

 - Test results.  
   - module_opt_unconstrained_test.ipynb  
   - module_opt_unconstrained_AD_test.ipynb  

---------------------------------------------------------------------------------------------------

Constrained Optimization Solver  
1. Quadratic Penalty Method(QPM) : qpm(f, ce, ci, x0, inner_opt, tol)  
2. Augmented Lagrangian Method(ALM) : alm(f, ce, ci, x0, inner_opt, tol)  
   1) ALM builds up unconstrained opt problem based on augmented Lagrangian function of original constrained opt problem at each k-th iteration
   2) Each unconstrained opt problem is solved using unconstrained optimizer 
4. Sequential Quadratic Programming(with ALM) : sqp(f, ce, ci, x0, maxiter=100, inner_opt=3, tol=1e-6, tol_inter=1e-4)  
   1) SQP loop builds up Quadratic Programming subproblem(QPk) at each k-th iteration  
   2) QPk is solved using alm4sqp(f, ce, ci, x0, lmbda0, nu0, inner_opt, tol)  
  
 - Test results.  
   - sqp_test_important.ipynb
   - assignment3.ipynb
   - assignment3_AD.ipynb
   - assignment5_*.ipynb ... refer to the assignment_5 pdf file.  

