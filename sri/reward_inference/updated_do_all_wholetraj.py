import os

models_dir = input('models_dir> ')
env_name = None
auto_env_names = ['spaceinvaders', 'miner', 'dodgeball', 'gridworld']
for potential_env_name in auto_env_names:
    if potential_env_name in models_dir:
        env_name = potential_env_name
        print("Automatically detected env_name", env_name)
if env_name is None:
    env_name = input('env_name> ')

if env_name == 'spaceinvaders' or env_name == 'gridworld':
    for npos in range(4):
        for nneu in range(4-npos):
            nneg = 3-npos-nneu

            st = ""
            sts = ""
            for i in range(npos):
                st += "positive_"
                sts += "+"
            for i in range(nneu):
                st += "neutral_"
                sts += "0"
            for i in range(nneg):
                st += "negative_"
                sts += "-"
            st = st[:-1]

            if os.system("python3.6 reward_alignment_verification_whole_trajectory.py " + sts + " " + env_name + " " + models_dir) != 0:
                import sys
                sys.exit()
else:
    for first in ['+', '0', '-']:
        for second in ['+', '0', '-']:
            for third in ['+', '0', '-']:
                sts = first + second + third
                if os.system("python3.6 reward_alignment_verification_whole_trajectory.py " + sts + " " + env_name + " " + models_dir) != 0:
                    import sys
                    sys.exit()
