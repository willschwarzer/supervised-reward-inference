print("NOTE run with python3.6 on base conda environment in october_miniconda_install")
import numpy as np

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import stable_baselines3
import random
import gym
import procgen
import os
import os.path
#os.mkdir('models_dodgeball')
#from new_block_env import *

transformer_dimensions = 256

class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len=251):
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
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=transformer_dimensions, nhead=16, batch_first=True)
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=1)
        self.avgpool = nn.AvgPool2d(2)
        self.conv = nn.Conv2d(3, 10, 5, stride=2)
        self.relu0 = nn.LeakyReLU()
        self.linear1 = nn.Linear(transformer_dimensions, 50)
        self.relu = nn.LeakyReLU()
        self.linear2 = nn.Linear(50, 12)

        self.conv1 = nn.Conv2d(3, 16, 7, stride=3)
        self.conv2 = nn.Conv2d(16, 16, 5, stride=2)
        self.conv3 = nn.Conv2d(16, 16, 3, stride=1)
        self.conv4 = nn.Conv2d(16, 16, 3, stride=1)

    def forward(self, states):
        a = states.permute(0, 1, 4, 2, 3)
        assert(a.shape[1] == 250)
        num_input = a.shape[0]
        x = a.view(num_input * 250, *(a.shape[2:]))
        x = x / 256 - 0.5
        x = self.conv1(x)
        x = F.leaky_relu(x)
        x = self.conv2(x)
        x = F.leaky_relu(x)
        x = self.conv3(x)
        x = F.leaky_relu(x)
        x = self.conv4(x)
        x = F.leaky_relu(x)
        #blurred = self.avgpool(conv_input)
        #x = self.conv(blurred)
        #x = self.relu0(x)
        x = x.reshape(num_input, 250, -1)
        x = self.positional_encoder(x)
        x = self.transformer(x)
        assert(x.shape[1] == 250)
        assert(x.shape[2] == transformer_dimensions)
        x = torch.mean(x, dim=1)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = x.view(-1, 3, 3)
        return x


class MlpNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(250*64*64*3, 20)
        self.relu = nn.LeakyReLU()
        self.linear2 = nn.Linear(20, 9)

    def forward(self, states):
        x = states.view(states.shape[0], 250*64*64*3)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = x.view(-1, 3, 3)
        return x

net = MlpNet().cuda()
optimizer = torch.optim.Adam(net.parameters())

def num_to_str(x):
    if x == -1:
        return "negative"
    elif x == 0:
        return "neutral"
    elif x == 1:
        return "positive"


def generate_training_batch():
    #return np.load('examples_dodgeball_temp.npy', allow_pickle=True)
    start_time = time.time()
    examples = []
    examples_per_type = 2
    num_models = 81
    first_time_save = True
    exampleno = 0
    for v1 in [-1, 0, 1]:
        for v2 in [-1, 0, 1]:
            for v3 in [-1, 0, 1]:
                listt = [v1, v2, v3]
                st = "_".join([num_to_str(x) for x in listt])
                path = "models_dodgeballnew/" + st

                print("Loading", path)
                env = gym.make('procgen-dodgeball-v0', extra_info=(','.join([str(x) for x in listt])))
                model = stable_baselines3.PPO.load(path, env)

                print("Done")
                for _ in range(examples_per_type):
                    example_obs = []
                    obs = env.reset()
                    for i in range(250):
                        action, _states = model.predict(obs)
                        obs, rewards, dones, info = env.step(action)
                        example_obs.append(obs) #env.render(mode='rgb').flatten())
                    example = {"key": listt, "data": np.array(example_obs), "info": info}
                    examples.append(example)
                    current_time = time.time()
                    estimated_total_time = (current_time-start_time) * examples_per_type * num_models / len(examples)
                    time_remaining = start_time + estimated_total_time - time.time()
                    print("\rExamples:", len(examples), "elapsed:", int(100*(current_time-start_time))/100.0, "time remaining:", int(time_remaining/60*100)/100, "mins", end="")
                print("")
                #if len(examples) >= 5000:
                #    np.save("examples_dodgeball/examples_dodgeball_" + str(exampleno) + ".npy", examples)
                #    exampleno += 1
                #    examples = []
    #np.sa(, examples)
    return examples

#np.save("examples_dodgeball/examples_dodgeball_" + str(exampleno) + ".npy", examples)

#for npos in range(5):
#  for nneu in range(5-npos):
#    nneg = 4-npos-nneu
#
#    listt = []
#    st = ""
#    for i in range(npos):
#      listt.append(1)#BlockType.POSITIVE)
#      st += "positive_"
#    for i in range(nneu):
#      listt.append(0)#BlockType.NEUTRAL)
#      st += "neutral_"
#    for i in range(nneg):
#      listt.append(-1)#BlockType.NEGATIVE)
#      st += "negative_"
#    st = st[:-1]
max_batch_size = 32

def pn(x):
    return (str(x) + (" " * 6))[0:6]

loss_func = nn.CrossEntropyLoss()

for epoch in range(50000):
    training = generate_training_batch()
    print("")
    print("training batch size:", len(training))
    for subepoch in range(10):
        print("subepoch", subepoch)
        random.shuffle(training)
        avg_loss = 0
        n_correct = 0
        avg_gradient_magnitude = 0
        n_samples = 0
        num_batches = math.ceil(len(training)/max_batch_size)
        for counter, batchno in enumerate(range(num_batches)):
            batch_start = batchno * max_batch_size
            batch_end = min(len(training), (batchno+1)*max_batch_size)
            batch_size = batch_end - batch_start
            batch = training[batch_start:batch_start+max_batch_size] # TODO replace with batch_end
            assert(batch_size == len(batch))

            optimizer.zero_grad()
            data = torch.stack([torch.Tensor(example["data"]).cuda() for example in batch]).float() / 255.0 * 2 - 1
            label = (torch.Tensor([[x+1 for x in example["key"]] for example in batch]).cuda().long())
            prediction = net.forward(data)
            loss = loss_func(prediction, label)
            avg_loss += loss.item() * batch_size
            n_correct += (torch.argmax(prediction, dim=1) == label).sum().item()
            if subepoch == 99:
                import code
                code.interact(local=locals())
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
                print("\r", n_samples, "/", len(training), "- epoch", epoch, "- avg loss", pn(avg_loss/n_samples), "- acc", pn(n_correct/(3*n_samples)), end="")

        avg_loss /= len(training)
        n_correct /= (3 * len(training))
        torch.save(net.state_dict(), "model_transformer_dodgeball.parameters")
        print("")
