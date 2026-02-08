#%%
import torch as th
from alg_zoo import zoo_2nd_argmax, zoo_argmedian, zoo_median
import numpy as np

hidden_size = 2
seq_len = 2
simple_model = zoo_median(hidden_size, seq_len)

#%%
x = th.tensor([[0, 10]], dtype=th.float32, device=simple_model.device)  # (batch=1, seq_len=2)

with th.no_grad():
    y = simple_model(x)          # or simple_model.forward(x)
print(simple_model)
print(y)
# %%
print("linear output matrix W^{h0}")
print(simple_model.linear.weight)
print("input matrix W^{hi}")
print(simple_model.rnn._parameters["weight_ih_l0"])
print("hidden matrix W^{hh}")
print(simple_model.rnn._parameters["weight_hh_l0"])
# %%
possible_hidden_matricies = [key for key in simple_model.rnn._parameters]
print("possible hidden matricies: f{possible_hidden_matricies}")

# %%
W_h0 = simple_model.linear.weight.detach().numpy()[0]
W_hi = simple_model.rnn._parameters["weight_ih_l0"].detach().numpy()
print(W_h0)
print(W_hi)

for index in range(len(W_h0)):
    # diff = abs(W_h0[index] - W_hi[index])
    sqrt_5 = np.sqrt(0.5)
    W_h0_diff = sqrt_5 - abs(W_h0[index])
    W_hi_diff = sqrt_5 - abs(W_hi[index])

    print(f"for {index} index: W_h0_diff:{W_h0_diff}, W_hi_diff:{W_hi_diff}")

# %%
