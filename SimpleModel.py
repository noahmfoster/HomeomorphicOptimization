import torch
import torch.nn as nn
from Layers import AnnealedLinear

class AnnealedModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, hidden_layers, bias=True, softmax=False):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers
        self.softmax = softmax

        self.raw_torch = False

        self.training = True
        self.temperature = torch.tensor(1.0, requires_grad=False)

        self.layers = nn.ModuleList()
        self.layers.append(AnnealedLinear(input_size, hidden_size, bias))
        for _ in range(hidden_layers):
            self.layers.append(AnnealedLinear(hidden_size, hidden_size, bias))
        self.layers.append(AnnealedLinear(hidden_size, output_size, bias))
        
    def forward(self, x):
        for layer in self.layers[:-1]:
            x = torch.relu(layer(x))
            x = torch.nn.functional.layer_norm(x, x.shape[1:])
        return self.layers[-1](x).softmax(dim=-1) if self.softmax else self.layers[-1](x)
    
    def set_temperature(self, temperature):
        self.temperature = temperature
        for layer in self.layers:
            layer.temperature = temperature
    
    def compute_l0(self):
        l0 = 0
        for layer in self.layers:
            l0 += layer.compute_l0()
        return l0
    
    def compute_weight(self):
        for layer in self.layers:
            layer.compute_weight()
            
    def get_num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def train(self):
        self.training = True
        for layer in self.layers:
            layer.training = True

    def eval(self):
        self.training = False
        for layer in self.layers:
            layer.training = False
        
        if not self.raw_torch: self.compute_weight()

    def to_normal_model(self):
        for i, layer in enumerate(self.layers):
            l_weight = torch.nn.Parameter(layer.weight)
            l_bias = torch.nn.Parameter(layer.bias)
            new_linear = torch.nn.Linear(layer.in_features, layer.out_features, bias=layer.bias is not None)
            new_linear.weight = l_weight
            new_linear.bias = l_bias
            self.layers[i] = new_linear
        self.raw_torch = True
        
    
