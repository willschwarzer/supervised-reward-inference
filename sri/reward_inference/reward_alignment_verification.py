from new_block_env import *
import stable_baselines3
import numpy as np
import os.path
import copy
import sys

models_dir = "models"

############################################################
#                    Get block types                       #
############################################################
char_to_block_type_info = {
    '+': ('positive', BlockType.POSITIVE),
    '-': ('negative', BlockType.NEGATIVE),
    '0': ('neutral', BlockType.NEUTRAL)
}
if len(sys.argv) >= 2:
    block_types_string = sys.argv[1]
else:
    block_types_string = input("Enter block types string > ")
assert(len(block_types_string) == 3)
assert(all([x in char_to_block_type_info.keys() for x in block_types_string]))
block_types = [char_to_block_type_info[x][1] for x in block_types_string]
block_types_strings = [str(char_to_block_type_info[x][0]) for x in block_types_string]
print('Block types:', ", ".join(block_types_strings))

############################################################
#                    Make environment                      #
############################################################
env = NewBlockEnv(block_types)

############################################################
#                    Load model                            #
############################################################
model_name_string = "_".join(block_types_strings) + ".zip"
model_path = models_dir + os.path.sep + model_name_string
print('Model path:', model_path)
assert(os.path.exists(model_path))
model = stable_baselines3.PPO.load(model_path, env)

############################################################
#                 Feature vector computations              #
############################################################
def compute_discounted_feature_vector_given_action(policy, environment_original, action, discount_factor = 1):
    environment = copy.deepcopy(environment_original)
    feature_counts = np.zeros_like(environment._get_feature_counts())
    obs, _, done, _ = environment.step(action)
    current_discount = 1
    while not done:
        feature_counts += environment._get_feature_counts() * current_discount
        current_discount *= discount_factor
        action, _ = policy.predict(obs)
        obs, _, done, _ = environment.step(action)
    return feature_counts

def compute_avg_discounted_feature_vector_given_action(policy, environment_original, action, discount_factor = 1, n_samples = 20):
    return sum([compute_discounted_feature_vector_given_action(policy, environment_original, action, discount_factor=discount_factor) for _ in range(n_samples)]) / n_samples

############################################################
#                 Get all the half planes                  #
############################################################

plane_normal_vectors_via_highest_return = []
plane_normal_vectors_via_det_action = []
obs = env.reset()
done = False
counter = 0 # just for the printout
while not done:
    print("\r", counter, "/", 150, end="")
    counter += 1
    #det_action, _ = model.predict(obs, deterministic=True)
    real_action, _ = model.predict(obs)
    det_action, _ = model.predict(obs, deterministic=True)
    action_counts = {}
    for action in Action:
        action_counts[action] = compute_avg_discounted_feature_vector_given_action(model, env, action)
    highest_expected_return = max(action_counts, key=lambda x: env.expected_return(action_counts[x]))
    for action in Action:
        if action is not highest_expected_return:
            plane_normal_vectors_via_highest_return.append(action_counts[highest_expected_return] - action_counts[action])
        if action is not det_action:
            plane_normal_vectors_via_det_action.append(action_counts[det_action] - action_counts[action])
    obs, reward, done, info = env.step(real_action)

    with open("planedata2/normal_vectors_via_highest_return_" + "_".join(block_types_strings) + ".txt", "w") as f:
        f.write(" ".join([str(x) for x in env.get_reward_weights()]) + "\n")
        f.write("\n".join([" ".join([str(x) for x in normal_vector]) for normal_vector in plane_normal_vectors_via_highest_return]))

    #with open("plane_normal_vectors_via_det_action.txt", "w") as f:
    #    f.write(" ".join([str(x) for x in env.get_reward_weights()]) + "\n")
    #    f.write("\n".join([" ".join([str(x) for x in normal_vector]) for normal_vector in plane_normal_vectors_via_det_action]))

print("\rDone")
