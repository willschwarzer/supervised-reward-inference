from new_block_env_regression import *
import stable_baselines3
import math
import torch
import torch.nn as nn
from functools import reduce
import torch.nn.functional as F
import numpy as np
import sys

print("WARNING not saving model")
#print("WARNING not printing intermediates")

model_path = sys.argv[1]
network_type = sys.argv[2]
method = sys.argv[3]
print('no_examples', model_path)
print('network_type', network_type)
print('method', method)

def label_from_example(example):
    ret = []
    for i in example["rgb_decode"][1:]:
        if i == BlockType.NEGATIVE:
            ret.append(-1.)
        elif i == BlockType.POSITIVE:
            ret.append(1.)
        elif i == BlockType.NEUTRAL:
            ret.append(0.)
        else:
            raise RuntimeError("Unknown block type")
    ret = np.array(ret)
    if abs(np.linalg.norm(ret)) > 0:
        ret /= np.linalg.norm(ret)
    return ret

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

class TransformerNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.positional_encoder = PositionalEncoder(transformer_dimensions)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=transformer_dimensions, nhead=10, batch_first=True)
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=1)
        self.linear1 = nn.Linear(100, 50)
        self.relu = nn.LeakyReLU()
        self.linear2 = nn.Linear(50, 3)

    def forward(self, states):
        x = self.positional_encoder(states)
        x = self.transformer(x)
        assert(x.shape[1] == 150)
        assert(x.shape[2] == 100)
        x = torch.mean(x, dim=1)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = x.view(-1, 3)
        return x

class LSTMNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(100, 32, 2, batch_first=True)
        self.linear1 = nn.Linear(32, 16)
        self.relu = nn.LeakyReLU()
        self.linear2 = nn.Linear(16, 9)

    def forward(self, states):
        x, _ = self.lstm(states)
        batch_size = states.shape[0]
        x = x[:, -1, :] #x.contiguous().view(batch_size, -1, 32)[-1]
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = x.view(-1, 3, 3)
        return x

class MlpNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(150*100, 200)
        self.relu = nn.LeakyReLU()
        self.linear2 = nn.Linear(200, 9)

    def forward(self, states):
        x = states.view(states.shape[0], 150*100)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = x.view(-1, 3, 3)
        return x

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 5, 5)
        self.relu1 = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(5, 3, 5, stride=2)
        self.relu2 = nn.LeakyReLU()
        self.conv3 = nn.Conv2d(3, 3, 5, stride=3)
        self.relu3 = nn.LeakyReLU()
        self.linear1 = nn.Linear(966, 100)
        self.relu4 = nn.LeakyReLU()
        self.linear2 = nn.Linear(100, 9)

    def forward(self, states):
        x = self.conv1(states.unsqueeze(dim=1))
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = x.view(x.shape[0], reduce(lambda a, b: a*b, x.shape[1:]))
        x = self.linear1(x)
        x = self.relu4(x)
        x = self.linear2(x)
        x = x.view(-1, 3, 3)
        return x

examples = np.load("examples_100000.npy", allow_pickle=True)
print("Found", len(examples), "examples.")

block_types = [BlockType.NEGATIVE, BlockType.POSITIVE, BlockType.NEUTRAL]

if network_type == 'lstm':
    net = LSTMNet().cuda()
elif network_type == 'transformer':
    net = TransformerNet().cuda()
elif network_type == 'mlp':
    net = MlpNet().cuda()
elif network_type == 'cnn':
    net = ConvNet().cuda()
net.load_state_dict(torch.load(model_path))

optimizer = torch.optim.Adam(net.parameters())
random.seed(5)
random.shuffle(examples)

# print("WARNING: Using direct training examples rather than whole set!")

# This is the good way!
training_split = int(0.8 * len(examples))

# Bad way
# training_split = no_examples

training = examples[0:training_split]
validation = examples[training_split:]
print("Num training", len(training))
print("Num validation", len(validation))
loss_func = nn.MSELoss()

max_batch_size = 1

def pn(x):
    return (str(x) + (" " * 6))[0:6]

import processed_plane_loader
planeauthority = processed_plane_loader.load_environment('gridworld')

total_reward = 0
total_alt_reward = 0
n_reward_samples = 0
for epoch in range(50):
    avg_loss = 0
    #n_correct = 0
    avg_gradient_magnitude = 0
    n_samples = 0
    num_batches = math.ceil(len(training)/max_batch_size)

    for counter, batchno in enumerate(range(num_batches)):
        if counter != 13:
            continue
        batch_start = batchno * max_batch_size
        batch_end = min(len(training), (batchno+1)*max_batch_size)
        batch_size = batch_end - batch_start
        batch = training[batch_start:batch_start+max_batch_size] # TODO replace with batch_end
        assert(batch_size == len(batch))

        optimizer.zero_grad()
        data = F.one_hot(torch.Tensor([example["data"] for example in batch]).cuda().long(), num_classes=4).view(-1, 150, 100).float()
        prediction = net.forward(data)[0]
        label = torch.Tensor([label_from_example(example) for example in batch]).cpu()[0]
        distance_to_arp = np.linalg.norm(prediction.detach().cpu().numpy() - planeauthority.comparison_point(prediction.detach().cpu().numpy(), label.detach().cpu().numpy(), mode='arp'))
        print("Distance to arp", distance_to_arp)
        alt_reward_label = [x.detach().cpu().item() for x in label]

        block_types = [BlockType.POSITIVE if x.item() > 0 else BlockType.NEGATIVE for x in prediction]
        block_powers = [abs(x.item()) for x in prediction]
        block_powers_real = [x.item() for x in prediction]
        env = NewBlockEnv(block_types, block_powers)
        agent = stable_baselines3.PPO("MlpPolicy", env, verbose=0)
        agent.learn(total_timesteps=100000)
        for _ in range(100):
            obs = env.reset()
            action = agent.predict(obs)
            for i in range(150):
                obs, reward, done, info = env.step(action, alt_reward=alt_reward_label)
                total_reward += reward
                total_alt_reward += info['alt_reward']
                n_reward_samples += 1
                action = agent.predict(obs)
            print(counter, total_reward/n_reward_samples, total_alt_reward/n_reward_samples, n_reward_samples, distance_to_arp)
