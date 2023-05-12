import torch
import torch.nn as nn
import torch.nn.functional as F

def double_sigmoid(x,y, temperature=1):
    sigmoid = torch.sigmoid(x * temperature) + ( 2 * torch.sigmoid(y * temperature) ) - 1
    return sigmoid

def hard_double_sigmoid(x,y):
    sigmoid = (x > 0).float() + ( 2 * (y > 0).float() ) - 1
    return sigmoid


class AnnealedLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.has_bias = bias
        self.temperature = torch.tensor(1.0, requires_grad=False)
        self.training = True

        self.pre_sigmoid_weight = nn.Parameter(torch.randn(2,out_features, in_features) * 3, requires_grad=True)

        if bias:
            self.pre_sigmoid_bias = nn.Parameter(torch.randn(2, out_features) * 3, requires_grad=True)


    def compute_weight(self):
        if self.training:
            self.weight = double_sigmoid(self.pre_sigmoid_weight[0], self.pre_sigmoid_weight[1], self.temperature)
            if self.has_bias:
                self.bias = double_sigmoid(self.pre_sigmoid_bias[0], self.pre_sigmoid_bias[1], self.temperature)

        else:
            self.weight = hard_double_sigmoid(self.pre_sigmoid_weight[0], self.pre_sigmoid_weight[1])
            if self.has_bias:
                self.bias = hard_double_sigmoid(self.pre_sigmoid_bias[0], self.pre_sigmoid_bias[1])

    def forward(self, x):
        self.compute_weight()
        if self.has_bias:
            return F.linear(x, self.weight, self.bias)
        else:
            return F.linear(x, self.weight) 
        
    def compute_l0(self):
        # double_sigmoid(x,y) = 0 when x= infty, y = -infty so 
        # l0(x,y) approxeq sigmoid(-x) + sigmoid(y)
        
        weight_y_minus_x = self.pre_sigmoid_weight[1] - self.pre_sigmoid_weight[0]
        weight_l0 = torch.sigmoid(weight_y_minus_x).sum()

        if self.has_bias:
            bias_y_minus_x = self.pre_sigmoid_bias[1] - self.pre_sigmoid_bias[0] 
            bias_l0 = torch.sigmoid(bias_y_minus_x).sum()
        else:
            bias_l0 = 0
        return (weight_l0 + bias_l0) 
