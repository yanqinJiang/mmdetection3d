import torch
from torch.autograd import Function
import torch.nn as nn

class GradReverse(Function):

    @ staticmethod
    def forward(ctx, x, lambd, **kwargs: None):
        #　其实就是传入dict{'lambd' = lambd}
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, *grad_output):
        # 传入的是tuple，我们只需要第一个
        return grad_output[0] * -ctx.lambd, None

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.Linear1 = nn.Sequential()
        self.Linear1.add_module('Linear_1', nn.Linear(1,1))
        self.Linear2 = nn.Sequential()
        self.Linear2.add_module('Linear_2', nn.Linear(1, 1))
        self.Linear3 = nn.Sequential()
        self.Linear3.add_module('Linear_3', nn.Linear(1,1))

    def forward(self,x):
        x = self.Linear1(x)
        x = self.Linear2(x)
        y = GradReverse.apply(x, 1.0)
        x = self.Linear3(x)
        y = self.Linear3(y)

        return x, y


if __name__ == '__main__':
    x_0 = torch.tensor([[1.]], requires_grad=True)

    net = Net()
    x, y = net(x_0)
    x.backward(retain_graph=True)
    for name, params in net.named_parameters():
        print('name:{}, params.grad{}'.format(name, params.grad))
    net.zero_grad()
    print('reverse grad')
    y.backward()
    for name, params in net.named_parameters():
        print('name:{}, params.grad{}'.format(name, params.grad))