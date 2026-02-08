import torch as th
from alg_zoo import zoo_median
import numpy as np

hidden_size = 2
seq_len = 3
simple_model = zoo_median(hidden_size, seq_len)

x = th.tensor([[0, -1, -2]], dtype=th.float32, device=simple_model.device)

with th.no_grad():
    y = simple_model(x)          
print(simple_model)
print(y)

print("input matrix W^{hi}")
print(simple_model.rnn._parameters["weight_ih_l0"])
print("hidden matrix W^{hh}")
print(simple_model.rnn._parameters["weight_hh_l0"])
print("linear output matrix W^{h0}")
print(simple_model.linear.weight)