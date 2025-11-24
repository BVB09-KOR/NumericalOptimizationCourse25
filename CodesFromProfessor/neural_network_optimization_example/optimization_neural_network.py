import matplotlib.pyplot as plt
import torch
import numpy as np
from train_objfunc import get_trained_objfunc
from constraints import constraint_1, constraint_2
from objective_function import obj_func

# Modified implementation of: Chen J, Liu Y. 2023 Neural optimization machine: a neural network approach for optimization and its application in additive manufacturing with physics-guided learning. Phil. Trans. R. Soc. A.

# Define the neural network for optimization problem(우리가 앞서서 이미 구한 NN surrogate model fr에 대한 최적화를 수행할 수 있는 NN 클래스 정의할거임)
class Net_optimizationproblem(torch.nn.Module):
    def __init__(self,f_neural_net, constraint):
        super(Net_optimizationproblem, self).__init__()
        self.f_neural_net = f_neural_net # NN of Objective function(앞서서 우리가 미리 학습시킨 NN surrogate model fr)
        for param in self.f_neural_net.parameters(): # Keep the NN objective function weights and biases constant ... 기학습된 surrogate 모델 fr의 weight, bias는 fr 자체를 최적화하는 NN 학습(즉 fr 최적화) 과정에서 고정시킨다(이거 안 하면 fr의 W, B도 최적화 과정 중 업데이트되서 개판된다).
            param.requires_grad = False # Does not compute gradients wrt weights and biases of the objective function.
        self.constraint = constraint # forward 함수(즉 NN이 최적화할 함수)가 constraint를 penalty term으로 반영한 함수로 쓰일 거기 때문에 미리 constraint를 속성으로 부여해준다.
        self.linear = torch.nn.Linear(2,2) # 학습시킬 레이어는 인풋 아웃풋 레이어가 전부. 아웃풋 레이어는 fr에 대한 input으로 사용될 x1, x2를 내보내야 되기에 당연히 2개 노드를 가지고 인풋 레이어의 노드 또한 2개로(우리 맘대로) 설정.
        torch.nn.init.eye_(self.linear.weight)
        torch.nn.init.zeros_(self.linear.bias)


    def forward(self, x): # 최적화시킬 함수(fr + penalty(constraint)) 정의
        x = self.linear(x)
        f_val = self.f_neural_net(x).squeeze()
        #f_val = obj_func(x); # Can also use the analytical objective function.
        constraints_val = self.constraint(x)

        output = f_val + 10*(torch.relu(constraints_val) + torch.relu(-constraints_val)) # penalty method의 form을 띄기에 constraint를 objective function에 결합
        return output

#Custom loss function for optimization.
class CustomLoss(torch.nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, predictions): # NN의 output으로 나오는 "fr + penalty(constraint)" 값에 대해 optimization할 거니까, loss function을 MSE 따위로 하지 않고(애초에 데이터셋 따위도 없음. 이건 NN '훈련'이 아니기 때문) NN의 output value 그 자체로 설정
        return predictions # 즉 항등함수

# constrained optimization(using surrogate fr based on NN) using NN
def run_optimizer(objfunc_neural_net, constraint):
    optimization_problem = Net_optimizationproblem(objfunc_neural_net, constraint);
    loss_function = CustomLoss() # NN의 output value로 나오는 "fr + penalty(constraint)" 값에 대해 optimization을 할 거니까, loss function을 MSE 따위로 하지 않고 output value 그 자체로 설정
    optimizer = torch.optim.NAdam(optimization_problem.parameters(), lr=0.01);

    constant = torch.Tensor(1,2) # NN 모델 input layer에 넣어줄 constant value로, NN optimization 내내 변하지 않는 값
    constant[:,0] = 0.5 # NN 모델 input layer에 넣어줄 constant value로, NN optimization 내내 변하지 않는 값
    constant[:,1] = 0.5 # NN 모델 input layer에 넣어줄 constant value로, NN optimization 내내 변하지 않는 값
    for epoch in range(10000):
        optimizer.zero_grad() # Gradient of wrt weights and biases are set to zero.
        outputs = optimization_problem(constant) # constant를 넣었을 때 NN(fr + penalty(constraint))이 내뱉는 output
        loss = loss_function(outputs) # 계산된 output에 항등함수 적용해서, 받은 output 그대로 내뱉고 그것을 loss로 설정
        loss.backward() # 계산된 loss에 대해서 backward propagation with automatic differentiation. Compute d (Obj_fun) / d (weights and biases)
        optimizer.step() # Updates weights and biases with specified learning rate
        if epoch % 1000 == 0:
            print("Epoch {}: Loss = {}".format(epoch, loss.detach().numpy()))

    x_optimal = optimization_problem.linear(constant);
    print("Optimal point = ", x_optimal);
    return x_optimal;



print("===================================================================");
print("                 Training NN objective function                       ");
print("===================================================================");
objfunc_neuralnet = get_trained_objfunc();
print("===================================================================");
print('\n\n');
print("===================================================================");
print("                 Optimization using Neural Network                 ");
print("===================================================================");
print('\n');
print("===================================================================");
print("Running optimization for f(x,y) = (1-x)^2 + 100*(y-x^2)^2");
print("With constraint c1(x,y) = 1-x-y = 0");
print("===================================================================");
run_optimizer(objfunc_neuralnet, constraint_1);
print("===================================================================");
print('\n\n');
print("===================================================================");
print("Running optimization for f(x,y) = (1-x)^2 + 100*(y-x^2)^2");
print("With constraint c2(x,y) = 1-x^2-y^2 = 0");
print("===================================================================");
run_optimizer(objfunc_neuralnet, constraint_2);
print("===================================================================");

