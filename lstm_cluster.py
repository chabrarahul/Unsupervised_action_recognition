import torch
inputs = torch.from_numpy(new_axis)
inputs=inputs.type(torch.float)
input_size = 51    #(without object)
batch_size = 1
hidden_size = 100
num_layers = 1
input = new_axis
hidden = (torch.randn(num_layers, batch_size, hidden_size, dtype=torch.float),torch.randn(num_layers, batch_size, hidden_size, dtype=torch.float))
lstm = torch.nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias = True,
            batch_first = True)
print(inputs.dtype)
print(hidden[1].dtype)
out,hidden = lstm(inputs,hidden)
print(out.size())
skeleton_numpy = out.detach().numpy()
input_skeleton = np.squeeze(skeleton_numpy, axis=0)




