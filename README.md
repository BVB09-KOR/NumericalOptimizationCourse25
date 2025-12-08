## For Numerical Optimization Course 2025 Fall Semester  
### Gradient Based Multivariate Optimization(Unconstrained & Constrained)  

(Refer to module_opt.py and module_opt_AD.py)  
 - module_opt.py ... Finite Difference based Gradient Calculation  
 - module_opt_AD.py ... Automatic Difference(provided by PyTorch) based Gradient Calculation
   - You should provide two types of functions as input argument, one with only numpy operations and another with only torch operations when using each optimizer

---------------------------------------------------------------------------------------------------

Unconstrained Optimizer  
1. Steepest Descent Method  
2. Conjugate Gradient Method  
3. Newton's Method(only in module_opt.py)  
4. Quasi Newton's Method - BFGS

 - Test results.  
module_opt_unconstrained_test.ipynb  
module_opt_unconstrained_AD_test.ipynb  

---------------------------------------------------------------------------------------------------

Constrained Optimizer  
1. Quadratic Penalty Method(QPM)  
2. Augmented Lagrangian Method(ALM)
   1) ALM builds up unconstrained opt problem based on augmented Lagrangian function of original constrained opt problem at each k-th iteration
   2) Each unconstrained opt problem is solved using unconstrained optimizer 
4. Sequential Quadratic Programming(with ALM)
   1) SQP loop builds up Quadratic Programming subproblem(QPk) at each k-th iteration  
   2) QPk is solved using ALM  
  
 - Test results.  
sqp_test_important.ipynb  
assignment3.ipynb  
assignment3_AD.ipynb  
assignment5_*.ipynb ... refer to the assignment_5 pdf file.  

