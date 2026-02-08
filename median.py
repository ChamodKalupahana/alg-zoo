#%%
import torch as th
from alg_zoo import zoo_2nd_argmax, zoo_argmedian

hidden_size = 2
seq_len = 3
simple_model = zoo_argmedian(hidden_size, seq_len)

#%%
x = th.tensor([[0, 1, 2]], dtype=th.float32, device=simple_model.device)  # (batch=1, seq_len=2)

with th.no_grad():
    y = simple_model(x)          # or simple_model.forward(x)
print(simple_model)
print(y)
# %%
