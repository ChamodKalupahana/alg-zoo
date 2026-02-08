#%%
from alg_zoo import example_2nd_argmax, zoo_2nd_argmax
model = example_2nd_argmax()

#%%
hidden_size = 2
seq_len = 2

simple_model = zoo_2nd_argmax(hidden_size, seq_len)
print(simple_model)
# %%

# W^hh
print(simple_model.linear.weight)
simple_model.rnn._parameters["weight_ih_l0"]
# %%
import torch as th
from alg_zoo import zoo_2nd_argmax

hidden_size = 2
seq_len = 2
simple_model = zoo_2nd_argmax(hidden_size, seq_len)

x = th.tensor([[0, 1]], dtype=th.float32, device=simple_model.device)  # (batch=1, seq_len=2)

with th.no_grad():
    y = simple_model(x)          # or simple_model.forward(x)
print(y)



# %%
