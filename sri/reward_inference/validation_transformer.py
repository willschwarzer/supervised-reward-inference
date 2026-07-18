from new_block_env import *
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

transformer_dimensions = 100

class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len=151):
        super().__init__()
        self.d_model = d_model
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = \
                  math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = \
                  math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        with torch.no_grad():
            x = x * math.sqrt(self.d_model)
            seq_len = x.size(1)
            pe = self.pe[:, :seq_len]
            x = x + pe
            return x

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.positional_encoder = PositionalEncoder(transformer_dimensions)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=transformer_dimensions, nhead=10, batch_first=True)
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=1)
        self.linear1 = nn.Linear(100, 50)
        self.relu = nn.LeakyReLU()
        self.linear2 = nn.Linear(50, 9)

    def forward(self, states):
        x = self.positional_encoder(states)
        x = self.transformer(x)
        assert(x.shape[0] == 1)
        assert(x.shape[1] == 150)
        assert(x.shape[2] == 100)
        x = torch.mean(x, dim=1)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = x.view(-1, 3, 3)
        return x

examples = np.load("examples.npy", allow_pickle=True)
print("Found", len(examples), "examples.")

block_types = [BlockType.NEGATIVE, BlockType.POSITIVE, BlockType.NEUTRAL]

net = Net()
net.load_state_dict(torch.load('model_transformer2.parameters'))

optimizer = torch.optim.Adam(net.parameters())
random.seed(5)
random.shuffle(examples)
training_split = int(0.8 * len(examples))
training = examples[0:training_split]
validation = examples[training_split:]
loss_func = nn.CrossEntropyLoss()

with torch.no_grad():
    for epoch in range(50):
        random.shuffle(validation)
        avg_loss = 0
        n_correct = 0
        for example in validation:
            data = F.one_hot(torch.Tensor(example["data"]).long(), num_classes=4).view(-1, 100).unsqueeze(dim=0).float()
            label = (torch.Tensor(example["rgb_decode"][1:]).long()).unsqueeze(dim=0)
            prediction = net.forward(data)
            loss = loss_func(prediction, label)
            avg_loss += loss.item()
            n_correct += (torch.argmax(prediction, dim=1) == label).sum().item()
        avg_loss /= len(validation)
        print("Finished epoch", epoch, "- avg loss", avg_loss)
        print("Accuracy", n_correct / (3 * len(validation)))


"""
def convert_to_colors(rgb, array):
  top = array.max()
  empty = array == BlockType.EMPTY
  array[empty] = top+1
  for block_type_index, block_type in enumerate(block_types):
    num_this_type = (rgb == block_type).sum()
    if num_this_type == 0:
      continue
    random = np.random.rand(*array.shape)
    matching = array == block_type
    for it in range(num_this_type):
      lower_bound = it / num_this_type
      upper_bound = (it+1) / num_this_type
      selection = np.logical_and(np.logical_and(random >= lower_bound, random < upper_bound), matching)
      counter = 0
      for rgb_index, rgb_type in enumerate(rgb):
        if rgb_type == block_type:
          if counter == it:
            array[selection] = rgb_index + top + 2
            break
          counter += 1
  array -= top + 1
  return array

for red in block_types:
  for green in block_types:
    for blue in block_types:
      current_block_types = sorted([red, green, blue])
      for example in examples:
        if current_block_types == sorted(example['key']):
          convert_to_colors(np.array([red, green, blue]), np.array(example['data']))
        import code
        code.interact(local=locals())
"""
