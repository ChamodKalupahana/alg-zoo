#%%
import torch as th
from alg_zoo import zoo_2nd_argmax

hidden_size = 2
seq_len = 2
simple_model = zoo_2nd_argmax(hidden_size, seq_len)

x = th.tensor([[0, 1]], dtype=th.float32, device=simple_model.device)  # (batch=1, seq_len=2)

with th.no_grad():
    output = simple_model(x)          # or simple_model.forward(x)
print(output)

for t in range(seq_len):
    print(f"h_{t+1} = {output[0, t]}")  # batch=0, timestep=t
# %%

# nn.RNN expects (batch, seq, input_size) unless batch_first=False
# We need to know whether rnn.batch_first is True and what input_size is.
print("batch_first:", getattr(simple_model.rnn, "batch_first", None))

# If the RNN expects features, add a feature dim (input_size=1)
x_rnn = x.unsqueeze(-1)   # (B, T, 1)

with th.no_grad():
    rnn_out, hT = simple_model.rnn(x_rnn)   # rnn_out: (B,T,H) if batch_first=True
print("rnn_out shape:", rnn_out.shape)
print("hT shape:", hT.shape)

# hidden state at each timestep:
h_1 = rnn_out[:, 0, :]  # (B,H)
h_2 = rnn_out[:, 1, :]  # (B,H)
print(h_1, h_2)
# %%
