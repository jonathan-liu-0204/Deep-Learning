import numpy as np
import matplotlib.pyplot as plt

# Data generation
def generate_linear(n=100):
    pts = np.random.uniform(0, 1, (n, 2))
    inputs = []
    labels = []

    for pt in pts:
        inputs.append([pt[0], pt[1]])
        distance = (pt[0] - pt[1]) / 1.414

        if pt[0] > pt[1]:
            labels.append(0)
        else:
            labels.append(1)
        
    return np.array(inputs), np.array(labels).reshape(n, 1)


def generate_XOR_easy():
    inputs = []
    labels = []

    for i in range(11):
        inputs.append([0.1*i, 0.1*i])
        labels.append(0)

        if 0.1*i == 0.5:
            continue

        inputs.append([0.1*i, 1 - 0.1*i])
        labels.append(1)
    
    return np.array(inputs), np.array(labels).reshape(21, 1)

# ========================================================
# Showing data

def show_data(X, Y, T):
    n = len(X)
    plt.figure(figsize = (5*n, 5))
    
    for i, x, y, t in zip(range(n), X, Y, T):
        y = np.round(y)
        plt.subplot(1, n, i+1)
        plt.title(t, fontsize = 18)
        for i in range(x.shape[0]):
            if y[i] == 0:
                plt.plot(x[i][0], x[i][1], 'ro')
            else:
                plt.plot(x[i][0], x[i][1], 'bo')
    
    plt.show()
# ========================================================
# Generate data now

# x1, y1 = generate_linear(n=100)
# x2, y2 = generate_XOR_easy()
# show_data([x1,x2], [y1,y2], ['Linear Data', 'XOR Data'])

# ========================================================
# Showing result

def show_result(x, y, pred_y):
    plt.subplot(1, 2, 1)
    plt.title('Ground truth', fontsize = 18)
    for i in range(x.shape[0]):
        if y[i] == 0:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')

    plt.subplot(1, 2, 2)
    plt.title('Predict result', fontsize = 18)
    pred_y = np.round(pred_y)
    for i in range(x.shape[0]):
        # print('pred_y: ', pred_y[i])
        if pred_y[i] == 0:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')
    
    plt.show()

# ========================================================
# Generate data now

# x1, y1 = generate_linear(n=100)
# x2, y2 = generate_XOR_easy()
# show_data([x1,x2], [y1,y2], ['Linear Data', 'XOR Data'])

# ========================================================
# Sigmoid functions

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))
    # return x

def derivative_sigmoid(x):
    return np.multiply(x, 1.0 - x)
    # return x

# check the sigmoid function
# xs = np.linspace(-10, 10, 500)
# plt.plot(xs, sigmoid(xs), label = 'Sigmoid')
# plt.plot(xs, derivative_sigmoid(sigmoid(xs)), label = 'Derivative Sigmoid')
# plt.legend(loc='upper left', shadow=True)
# plt.show()

# ========================================================
# MSE - Loss Function

def mse_loss(y, yhat):
    return np.mean((y - yhat)**2)

def derivative_mse_loss(y, yhat):
    return (y - yhat)*(2 / y.shape[0])

# ========================================================
# Unit Calculator

class unit_calculation():
    def __init__(self, inputsize, outputsize):
        self.w = np.random.normal(0, 1, (inputsize + 1, outputsize))
    
    def forward(self, x):
        x = np.append(x, np.ones((x.shape[0], 1)), axis = 1)
        self.forward_grad = x
        self.y = sigmoid(np.matmul(x, self.w))
        # =================================================
        # without activation functions
        # self.y = np.dot(x, self.w)
        # =================================================
        return self.y

    def backward(self, der_c):
        self.backward_grad = np.multiply(derivative_sigmoid(self.y), der_c)
        # =================================================
        # without activation functions  
        # self.backward_grad = np.multiply(self.y, der_c)
        # self.backward_grad = np.array(self.backward_grad, dtype=np.float64)
        # =================================================
        return np.dot(self.backward_grad, self.w[ : -1].T)
    
    def update(self, learning_rate):
        self.gradient = np.matmul(self.forward_grad.T, self.backward_grad)
        self.w = self.w - learning_rate * self.gradient
        return self.gradient

# ========================================================
# Neural Network

class NN():
    # initialize the weight matrix
    def __init__(self, size, learning_rate):
        self.learning_rate = learning_rate
        self.layers = []

        for a, b in zip(size, (size[1:] + [0])):
            if (a+1)*b != 0:
                self.layers += [unit_calculation(a, b)]

    def forward(self, x):
        new_x = x
        for layer in self.layers:
            new_x = layer.forward(new_x)
        return new_x

    def backward(self, der_cost):
        new_der_cost = der_cost
        for layer in self.layers[::-1]:
            new_der_cost = layer.backward(new_der_cost)

    def update(self):
        gradients = []
        for layer in self.layers:
            gradients = gradients + [layer.update(self.learning_rate)]
        return gradients

# ========================================================
# testing

linear_nn = NN([2, 4, 4, 1], 0.1)
XOR_nn = NN([2, 4, 4, 1], 0.1)

epoches = 40000000
threshold = 0.005

linear_stop = False
XOR_stop = False

linear_x, linear_y = generate_linear()
XOR_x, XOR_y = generate_XOR_easy()
show_data([linear_x, XOR_x], [linear_y, XOR_y], ['Linear Data', 'XOR Data'])

# store data to draw the learning curve
linear_loss_data = []
xor_loss_data = []
epoch_data = []

for epoch in range(epoches):
    if not linear_stop:
        y1 = linear_nn.forward(linear_x)
        linear_loss = mse_loss(y1, linear_y)
        linear_nn.backward(derivative_mse_loss(y1, linear_y))
        linear_nn.update()

        if linear_loss < threshold:
            linear_stop = True
            print("Epoch: ", epoch)
            print("Linear goal accomplished...:)")

    if not XOR_stop:
        y2 = XOR_nn.forward(XOR_x)
        XOR_loss = mse_loss(y2, XOR_y)
        XOR_nn.backward(derivative_mse_loss(y2, XOR_y))
        XOR_nn.update()

        if XOR_loss < threshold:
            XOR_stop = True
            print("Epoch: ", epoch)
            print("XOR goal accomplished...:)")

    if epoch % 5000 == 0:
        print('epoch {:4d} linear loss : {:.4f} \t XOR loss : {:.4f}'.format(epoch, linear_loss, XOR_loss))

        linear_loss_data.append(linear_loss)
        xor_loss_data.append(XOR_loss)
        epoch_data.append(epoch)

    if linear_stop and XOR_stop:
        print('epoch {:4d} linear loss : {:.4f} \t XOR loss : {:.4f}'.format(epoch, linear_loss, XOR_loss))
        linear_loss_data.append(linear_loss)
        xor_loss_data.append(XOR_loss)
        epoch_data.append(epoch)
        break

# show the result comparison figure
show_result(linear_x, linear_y, y1)
show_result(XOR_x, XOR_y, y2)

print('\nLinear test loss : ', mse_loss(y1, linear_y))
print('Linear test result : \n', y1.round(3))

print('XOR test loss : ', mse_loss(y2, XOR_y))
print('XOR test result : \n', y2.round(4))

plt.plot(epoch_data, linear_loss_data, label = 'Linear')
plt.plot(epoch_data, xor_loss_data, label = 'XOR')
plt.legend(loc='upper left', shadow=True)
plt.title("Learning Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()
