import numpy as np
import torch
from objective_function import obj_func

# Neural network (NN) for objective function. 
# The NN has two inputs, three hidden layers and one output.
# torch.nn.Linear stores weights and biases, which are trained to minimize error with respect to the objective function.

class Net(torch.nn.Module): # 내가 훈련시키길 원하는 NN 모델에 대한 상세사항을 정의하는 클래스(torch.nn.Module 클래스 상속)를 이렇게 만들어줘야지만 torch로 NN 모델 학습 가능.
    def __init__(self):
        # torch.nn.Module(부모 클래스) __init__ 함수 상속하여 현재 정의하고 있는 자식 클래스(Net) __init__ 함수 실행 시 자동 실행
        super(Net, self).__init__()
        
        # Specify layers of the neural network.
        n_inputs = 2
        n_layer1 = 50
        n_layer2 = 100
        n_layer3 = 50
        n_outputs = 1
        
        # 각 레이어들 사이의 linear transformation 함수 정의(linear form으로 정의)
        self.hidden1 = torch.nn.Linear(n_inputs, n_layer1) # 2 inputs, n_layer1 neurons
        self.hidden2 = torch.nn.Linear(n_layer1, n_layer2) # n_layer1 inputs, n_layer2 neurons
        self.hidden3 = torch.nn.Linear(n_layer2, n_layer3) # n_layer2 inputs, n_layer3 neurons
        self.output_layer = torch.nn.Linear(n_layer3, n_outputs) # n_layer3 inputs, 1 output
        
        # Initialize weights and biases(상기에서 정의한 linear transformation 함수에서 쓰이는 Weight 및 Bias initialization).
        torch.nn.init.eye_(self.hidden1.weight)
        torch.nn.init.ones_(self.hidden1.bias)
        torch.nn.init.eye_(self.hidden2.weight)
        torch.nn.init.ones_(self.hidden2.bias)
        torch.nn.init.eye_(self.hidden3.weight)
        torch.nn.init.ones_(self.hidden3.bias)
        torch.nn.init.eye_(self.output_layer.weight)
        torch.nn.init.ones_(self.output_layer.bias)

    def forward(self, x): # NN 모델에서의 output evaluation 정의
        x = torch.relu(self.hidden1(x)) # hidden layer 1에서의 output(ReLU 함수 사용)
        x = torch.relu(self.hidden2(x)) # hidden layer 2에서의 output(ReLU 함수 사용)
        x = torch.relu(self.hidden3(x)) # hidden layer 3에서의 output(ReLU 함수 사용)
        x = self.output_layer(x) # output layer에서의 output(ReLU 함수 미사용)
        return x


def get_trained_objfunc(): # 상기에서 기정의된 Net 클래스 사용하여 obj_func에 대한 surrogate model(using NN) 생성하는 함수

    # Data to train the neural network.
    n_samples_1d = 20; # 1개 입력 변수 차원당 20개의 데이터 샘플을 훈련에 사용 (2개 입력 변수 사용 시 20 x 20 = 400개 데이터를 훈련에 사용)
    x_1d =  torch.linspace(0.0,1.2, n_samples_1d); # 
    x_samples,y_samples = np.meshgrid(x_1d,x_1d)
    xy_tensor = torch.from_numpy(np.concatenate((x_samples.reshape(n_samples_1d**2,1), y_samples.reshape(n_samples_1d**2,1)),axis=1))
    f_exact = obj_func(xy_tensor).unsqueeze(1); # 20 x 20 데이터 샘플에 대한 true function value evaluation

    f_neural_net = Net() # 함수 훈련시킬 NN 모델 initialize(상기에서 정의한 클래스 사용)
    loss_function = torch.nn.MSELoss() # Using squared L2 norm of the error(함수 훈련 시 사용할 loss function initialize ; Mean Square Error (1/n)Σ(fi-fi_r)^2).
    optimizer = torch.optim.Adam(f_neural_net.parameters(), lr=0.005)  # Adam optimizer initialize(loss function 최적화(즉 NN 모델 훈련)할 Adam optimizer initialize).

    print("Training neural network of objective function")
    for epoch in range(10000):
        optimizer.zero_grad() # 매 epoch에서의 gradient 누적 방지 위해서 매 epoch마다 0으로 초기화(안 하면 gradient가 누적되어 더해진다 ; 굳이 이렇게 더하는 게 디폴트인 이유는 RNN 내부 루프 훈련 시 이 과정이 필요하기 때문)
        outputs = f_neural_net(xy_tensor) # 데이터셋의 input을 훈련시키는 NN 모델에 집어넣었을 때의 예측 ouput
        loss = loss_function(outputs, f_exact) # 훈련 NN 모델의 loss 값
        loss.backward() # Backpropagation and computing GRADIENTS w.r.t. weights and biases(design variables).
        optimizer.step() # UPDATE weights and biases(design variables).

        if epoch % 500 == 0:
            print("Epoch {}: Loss = {}".format(epoch, loss.detach().numpy()))

    print("Finished training neural network of objective function");

    return f_neural_net