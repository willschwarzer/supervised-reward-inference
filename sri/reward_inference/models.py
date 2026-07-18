from sri.reward_inference.modules import *
from set_transformer.models import SetTransformer
import torch
import torch.nn as nn


class NonLinearNet(nn.Module):

    def __init__(
            self,
            demonstration_rep_dim,
            state_rep_dim,
            internal_tst_dim,
            state_hidden_size,
            final_hidden_size,
            demonstration_hidden_size,
            obs_size,  # for state encoder
            dem_obs_size,  # for demonstration encoder
            horizon,
            num_demonstration_layers,
            num_state_layers,
            dem_encoder_type='tst',  # tst stands for transformer->set transformer
            mlp=False,
            demonstration_sigmoid=False,
            ground_truth_phi=False,
            output_type='reward',
            action_dim=4,
            transformer_nhead=None,):

        super().__init__()
        assert (
            demonstration_rep_dim == state_rep_dim or 
            mlp or
            output_type == "goal"
        ), "Non-matching rep dims only implemented for mlp"
        # if direct_goal_inference:
        #     assert dem_encoder_type == 'tst', "Direct goal inference only implemented for TST!"
        if dem_encoder_type == 'lstm':
            self.demonstration_encoder = DemonstrationNetLSTM(
                dem_obs_size, num_demonstration_layers)
        elif dem_encoder_type == 'transformer':
            self.demonstration_encoder = DemonstrationNetTransformer(
                dem_obs_size, horizon, num_demonstration_layers,
                demonstration_hidden_size, demonstration_rep_dim,
                demonstration_sigmoid, transformer_nhead)
        elif dem_encoder_type == 'set_transformer':
            self.demonstration_encoder = SetTransformer(
                dem_obs_size * horizon, 1, demonstration_rep_dim)
        elif dem_encoder_type == 'tst':
            tst_dem_encoder = DemonstrationNetTransformer(
                dem_obs_size, horizon, num_demonstration_layers,
                demonstration_hidden_size, internal_tst_dim,
                nhead=transformer_nhead)
            tst_set_encoder = SetTransformer(internal_tst_dim, 1,
                                             demonstration_rep_dim)
            self.demonstration_encoder = TST(tst_dem_encoder, tst_set_encoder)
        else:
            raise ValueError("Invalid demonstration encoder type!")
        self.dem_encoder_type = dem_encoder_type
        # self.demonstration_encoder = SIDemonstrationNet()
        if output_type != "goal":
            self.state_encoder = StateNet(state_rep_dim, state_hidden_size,
                                      num_state_layers, obs_size)
        self.mlp = mlp
        self.horizon = horizon
        if mlp:
            if output_type == "reward":
                out_dim = 1
            elif output_type == "action":
                out_dim = action_dim
            else:
                raise ValueError(f"Cannot use MLP with output type {output_type}")
            self.final_layer = MLP(demonstration_rep_dim, state_rep_dim,
                                          final_hidden_size, out_dim)
        elif output_type != "goal":
            assert (demonstration_rep_dim == state_rep_dim)
            assert output_type == "reward", "Non-MLP not implemented for action inference! (there's a dot product in there)"
        assert not ground_truth_phi, "Ground truth phi not implemented anymore!"
        # self.ground_truth_phi = ground_truth_phi
        # if ground_truth_phi:
        #     self.MIN_OBJECT_DIST = 0.25
        #     assert output_type == "reward", "Ground truth phi only implemented for reward inference!"
        self.output_type = output_type

    def forward(self, demonstrations, states=None, weights=None):
        # demonstrations: (bsize, L, |S|) or (bsize, n, L, |S|)
        # weights ~= goal position
        # But I don't think that's implemented anymore
        assert weights is None, "Weights not implemented anymore!"
        if weights is None:
            if self.dem_encoder_type == "set_transformer":
                # set transformer is doing sequence modeling along the "n" dimension (although it presumably assumes batching)
                # so we need to flatten the last two dimensions
                dem_input = demonstrations.view(demonstrations.shape[0],
                                                demonstrations.shape[1], -1)
            else:
                dem_input = demonstrations
            # we don't need to do anything for TST because we defined it to take in the right shape
            # demonstration_rep = self.demonstration_encoder(dem_input).squeeze(
            #     dim=1)  # (bsize, rep_dim)
            demonstration_rep = self.demonstration_encoder(dem_input).squeeze()  # (bsize, rep_dim)
            # if self.direct_goal_inference:
            if self.output_type == "goal":
                return demonstration_rep
        else:
            demonstration_rep = weights.squeeze()  # (bsize, rep_dim)
        # states = demonstrations.view(-1, demonstrations.shape[-1]) # (bsize*L, |S|)
        # if not self.ground_truth_phi:
        state_rep = self.state_encoder(states)  # (bsize*L, rep_dim)
        # Reshape to flatten the last two dimensions
        state_rep_flat = state_rep.reshape(-1, state_rep.shape[-1])
        # else:
        #     state_rep = _get_reward_features_torch(
        #         states, self.MIN_OBJECT_DIST,
        #         True).cuda()  # (bsize, num_states, 5, 5)

        #     # this is technically unflattening, but we need the names to match
        #     state_rep_flat = state_rep.view(states.shape[0] * states.shape[1],
                                            # -1)
        # If the batch size was 1, unsqueeze demonstration_rep to make it 2D
        if demonstrations.shape[0] == 1:
            demonstration_rep = demonstration_rep.unsqueeze(0)
        # demonstration_rep_expanded = demonstration_rep.unsqueeze(1).expand(-1, self.horizon, -1) # (bsize, L, rep_dim)
        # demonstration_rep_flattened = demonstration_rep_expanded.reshape(-1, demonstration_rep_expanded.shape[-1]) # (bsize*L, rep_dim)
        # not sure why we were doing that; we should do num_states instead of L
        demonstration_rep_expanded = demonstration_rep.unsqueeze(1).expand(
            -1, states.shape[1], -1)
        demonstration_rep_flattened = demonstration_rep_expanded.reshape(
            -1, demonstration_rep_expanded.shape[-1])
        # also flatten the latter two dimensions of state_rep
        if self.mlp:
            state_rep_flattened = state_rep.reshape(-1, state_rep.shape[-1])
            out = self.final_layer(demonstration_rep_flattened,
                                       state_rep_flattened)
            # if self.output_type == "action":
            #     # need to softmax the action output
            #     out = torch.softmax(out, dim=-1)
        else:
            out = torch.sum(demonstration_rep_flattened * state_rep_flat,
                               dim=1)
        return out.view(states.shape[0], -1)
