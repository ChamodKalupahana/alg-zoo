import torch as th

from alg_zoo.architectures import DistRNN, OneLayerRNN, ScalarRNN

def handcrafted_median_model():
    model = ScalarRNN(hidden_size=2, seq_len=2, bias=False)
    tensor_to_set = th.tensor(0.5, dtype=model.dtype, device=model.device)
    sqrt_5 = th.sqrt(tensor_to_set)
    model.rnn.weight_ih_l0.data = th.tensor(
        [[sqrt_5], [-sqrt_5]], dtype=model.dtype, device=model.device
    )
    model.rnn.weight_hh_l0.data = th.tensor(
        [[1, -1], 
         [-1, 1]], dtype=model.dtype, device=model.device
    )
    model.linear.weight.data = th.tensor(
        [[sqrt_5, -sqrt_5]], dtype=model.dtype, device=model.device
    )
    return model

handcrafted_model = handcrafted_median_model()

x = th.tensor([[5, 10]], dtype=th.float32, device=handcrafted_model.device)  # (batch=1, seq_len=2)

with th.no_grad():
    y = handcrafted_model(x)          # or simple_model.forward(x)
print(handcrafted_model)
print(y)