# Optimization using Neural Network

Performs constrained optimization using neural network with PyTorch.  
To install pytorch: `pip install torch`  

## Description of files
Objective function and constraints are included in `objective_function.py` and `constraints.py`. File `optimization_neural_network.py` calls the function in `train_objfunc.py` to train the objective function and runs optimization using neural network.  
  
큰 호출 : optimization_neural_network.py CALLS train_objfunc.py & objective_function.py & constraints.py  
작은 호출 : train_objfunc.py CALLS objective_function.py  
작은 호출 ⊂ 큰 호출  

큰 호출(optimization_neural_network.py)이 하는 작업 : NN 사용하여 원래의 obj function에 대한 surrogate model(also from NN) 최적화 수행  
작은 호출(train_objfunc.py)이 하는 작업 : NN 사용하여 원래의 obj function에 대한 surrogate model 생성하는 함수 정의  
objective_function.py : 최적화하고자 하는 obj function 정의  
constraints.py : 최적화 시 고려해야 하는 constraints 정의  
