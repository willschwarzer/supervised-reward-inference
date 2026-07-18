from new_block_env import *
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys

no_examples = int(sys.argv[1])

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
        self.linear2 = nn.Linear(50, 3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, states):
        x = self.positional_encoder(states)
        x = self.transformer(x)
        assert(x.shape[1] == 150)
        assert(x.shape[2] == 100)
        x = torch.mean(x, dim=1)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        #x = x.view(-1, 3)
        x = self.sigmoid(x)
        return x

examples = np.load("examples_380000_regression.npy", allow_pickle=True)
print("Found", len(examples), "examples.")

block_types = [BlockType.NEGATIVE, BlockType.POSITIVE, BlockType.NEUTRAL]

net = Net().cuda()
#net.load_state_dict(torch.load("model_transformer_regression_380000.parameters"))

optimizer = torch.optim.Adam(net.parameters())
random.seed(5)
random.shuffle(examples)

print("WARNING: Using direct training examples rather than whole set!")

training_split = int(0.8 * len(examples))
# Bad way
#training_split = no_examples

training = examples[0:training_split]
validation = examples[training_split:]
print("Num training", len(training))
print("Num validation", len(validation))
loss_func = nn.MSELoss() #nn.CrossEntropyLoss()

max_batch_size = 32

def pn(x):
    return (str(x) + (" " * 6))[0:6]

def label_from_example(example):
    ret = []
    assert(len(example["block_powers"]) == len(example["block_types"]))
    for i in range(len(example["block_powers"])):
        if example["block_types"][i] == BlockType.NEGATIVE:
            ret.append(example["block_powers"][i] * -1)
        elif example["block_types"][i] == BlockType.POSITIVE:
            ret.append(example["block_powers"][i])
        elif example["block_types"][i] == BlockType.NEUTRAL:
            ret.append(0)
        else:
            raise RuntimeError("Unknown block type")
    return ret

differences_removeme = []

for epoch in range(50):
    random.shuffle(training)
    avg_loss = 0
    #n_correct = 0
    avg_gradient_magnitude = 0
    n_samples = 0
    num_batches = math.ceil(len(training)/max_batch_size)

    with open("differences_removeme.txt", "w") as f:
        f.write(",".join([str(x) for x in differences_removeme]))

    for counter, batchno in enumerate(range(num_batches)):
        batch_start = batchno * max_batch_size
        batch_end = min(len(training), (batchno+1)*max_batch_size)
        batch_size = batch_end - batch_start
        batch = training[batch_start:batch_start+max_batch_size] # TODO replace with batch_end
        assert(batch_size == len(batch))

        optimizer.zero_grad()
        data = F.one_hot(torch.Tensor([example["data"] for example in batch]).cuda().long(), num_classes=4).view(-1, 150, 100).float()
        label = torch.Tensor([label_from_example(example) for example in batch]).cuda() / 2 + 0.5 #(torch.Tensor([example["rgb_decode"][1:] for example in batch]).cuda().long())
        prediction = net.forward(data)
        loss = loss_func(prediction, label)

        difference_removeme = (prediction - label) * 2
        for i in range(prediction.shape[0]):
            difference_removeme_tuple = (difference_removeme[i][0].item(), difference_removeme[i][1].item(), difference_removeme[i][2].item())
            differences_removeme.append(difference_removeme_tuple[0])
            differences_removeme.append(difference_removeme_tuple[1])
            differences_removeme.append(difference_removeme_tuple[2])

        avg_loss += loss.item() * batch_size
        #n_correct += (torch.argmax(prediction, dim=1) == label).sum().item()
        n_samples += batch_size
        loss.backward()
        optimizer.step()
        #print("train", example["data"].sum())
        #for name, param in net.named_parameters():
        #  if param.grad is not None:
        #    avg_gradient_magnitude += torch.norm(param.grad)
        #print("\ravg grad mag", avg_gradient_magnitude/counter, end="")

        # UNCOMMENT TO PRINT INTERMEDIATES
        if counter > 0:
            print("\r", n_samples, "/", len(training), "- epoch", epoch, "- avg loss", pn(avg_loss/n_samples), end="")

    avg_loss /= len(training)
    #n_correct /= (3 * len(training))

    # UNCOMMENT TO SAVE MODEL
    torch.save(net.state_dict(), "model_transformer_regression_lastminute_gridworld.parameters")

    avg_training_loss = avg_loss
    #n_correct_training = n_correct
    max_strlen = 0
    avg_loss = 0
    #n_correct = 0
    n_samples = 0
    with torch.no_grad():
        num_batches = math.ceil(len(validation)/max_batch_size)
        for counter, batchno in enumerate(range(num_batches)):
            batch_start = batchno * max_batch_size
            batch_end = min(len(validation), (batchno+1)*max_batch_size)
            batch_size = batch_end - batch_start
            batch = validation[batch_start:batch_start+max_batch_size] # TODO replace with batch_end
            assert(batch_size == len(batch))

            #print("valid", example["data"].sum())
            data = F.one_hot(torch.Tensor([example["data"] for example in batch]).cuda().long(), num_classes=4).view(-1, 150, 100).float()
            label = (torch.Tensor([example["rgb_decode"][1:] for example in batch]).cuda().long()) / 2 + 0.5
            prediction = net.forward(data)
            loss = loss_func(prediction, label)
            avg_loss += loss.item() * batch_size
            #n_correct += (torch.argmax(prediction, dim=1) == label).sum().item()
            n_samples += batch_size

            if counter > 0:
                string_to_print = " ".join([str(x) for x in ["\rValid", n_samples, "/", len(validation), "- epoch", epoch, "- avg loss", pn(avg_training_loss), "- val loss", pn(avg_loss/n_samples)]])
                if len(string_to_print) > max_strlen:
                    max_strlen = len(string_to_print)
                print(string_to_print, end="")
        avg_loss /= len(validation)
        #n_correct /= (3 * len(validation))
        string_to_print = " ".join([str(x) for x in ["\rFinished epoch", epoch, "- avg loss", pn(avg_training_loss), "- val loss", pn(avg_loss)]])
        print(string_to_print + (" " * (max_strlen - len(string_to_print))))
    print("WARNING not saving")


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
