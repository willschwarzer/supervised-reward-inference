import argparse
import os
from pathlib import Path
import numpy as np
from tqdm import tqdm
import wandb
import torch
from sri.reward_inference.train import (
    train as train_reward_model,
    load_model,
    load_data,
)
from sri.reward_inference.IL_evaluation import train_bc, train_adversarial
from sri.rl.train import train as train_policy
from sri.rl.train import make_env
from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE

# no need to import hidden, you can just set env._partially_observable = True
from metaworld.policies.sawyer_push_v2_policy import SawyerPushV2Policy
from metaworld.policies.sawyer_reach_v2_policy import SawyerReachV2Policy
from metaworld.policies.sawyer_pick_place_v2_policy import SawyerPickPlaceV2Policy
from sri.utils import (
    namespace_to_dict,
    convert_rollouts_to_chai,
    get_dataset_name,
    update_wandb_with_namespaces_and_names,
    get_model_names,
    get_eval_results,
    load_config_with_defaults,
)
from sri.run_experiment import process_configs
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, DummyVecEnv
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.policies import ActorCriticPolicy
from sb3_contrib import TQC
import pandas as pd
import imageio
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
import ipdb
from tqdm import tqdm

plt.rcParams["font.family"] = "Times New Roman"

CACHE_DIR = Path("cache")
RESULTS_DIR = Path("results")
PLOTS_DIR = RESULTS_DIR / "plots"
TABLES_DIR = RESULTS_DIR / "tables"
LOGS_DIR = RESULTS_DIR / "logs"
INCLUDE_PEMIRL = False


def ensure_output_dirs():
    for path in (CACHE_DIR, PLOTS_DIR, TABLES_DIR, LOGS_DIR):
        path.mkdir(parents=True, exist_ok=True)


def _maybe_include_pemirl_baseline(config_paths):
    if INCLUDE_PEMIRL and "baselines/pemirl.yml" not in config_paths:
        return config_paths + ["baselines/pemirl.yml"]
    return config_paths


def _append_baseline_result(
    baseline_config,
    result,
    gail_results,
    airl_results,
    bc_results,
    pemirl_results,
):
    if "gail" in baseline_config.baselines:
        gail_results.append(result)
    elif "airl" in baseline_config.baselines:
        airl_results.append(result)
    elif "bc" in baseline_config.baselines:
        bc_results.append(result)
    elif "pemirl" in baseline_config.baselines:
        pemirl_results.append(result)
    else:
        raise ValueError(f"Unknown baseline: {baseline_config.baselines}")


def _optional_array(values):
    if values is None:
        return None
    arr = np.array(values)
    if arr.size == 0:
        return None
    return arr


def _load_optional_array(npz_data, key):
    if key not in npz_data.files:
        return None
    arr = npz_data[key]
    if arr.size == 0:
        return None
    return arr


def _save_optional_array(values):
    if values is None:
        return np.array([])
    return values



def get_pemirl_smoke_results(num_runs=1):
    """Collect tiny PEMIRL smoke eval metrics from W&B evaluation runs."""
    model_idxs = list(range(num_runs))
    smoke_n = int(os.environ.get("PEMIRL_SMOKE_N", "3"))
    smoke_horizon = int(os.environ.get("PEMIRL_SMOKE_HORIZON", "50"))
    smoke_bc_epochs = int(os.environ.get("PEMIRL_SMOKE_BC_EPOCHS", "50"))
    smoke_adv_its = int(os.environ.get("PEMIRL_SMOKE_ADV_ITS", "5000000"))

    general_config = load_config_with_defaults("general/default_eval.yml")
    train_config = load_config_with_defaults("train/n1_baselines.yml")
    dataset_config = load_config_with_defaults("datasets/default_reach.yml")
    obs_dataset_config = load_config_with_defaults("datasets/default_reach.yml")
    model_config = load_config_with_defaults("model/load_default.yml")
    rl_config = load_config_with_defaults("rl/tqc_quarter_legacy_paper_esr_5m.yml")
    inference_config = load_config_with_defaults("inference/reach_baselines_dummy.yml")
    baselines_config = load_config_with_defaults("baselines/pemirl.yml")

    # Keep this aligned with the smoke sbatch command so filters match exactly.
    inference_config.evaluation = True
    inference_config.baselines_only = True
    inference_config.include_actions = True
    inference_config.skip_all_inference = False
    inference_config.only_rl = False
    inference_config.episodes = 1
    inference_config.num_goals = 1
    inference_config.batch_size = 1
    inference_config.horizon = smoke_horizon
    inference_config.n = smoke_n
    baselines_config.bc_epochs = smoke_bc_epochs
    baselines_config.adv_its = smoke_adv_its

    process_configs(
        general_config,
        dataset_config,
        obs_dataset_config,
        model_config,
        train_config,
        rl_config,
        inference_config,
        baselines_config,
    )

    result = get_eval_results(
        general_config,
        model_config,
        train_config,
        dataset_config,
        obs_dataset_config,
        most_recent_first=True,
        model_idxs=model_idxs,
        rl=False,
        rl_config=rl_config,
        baseline=True,
        baselines_config=baselines_config,
        inference_config=inference_config,
    )
    return np.array(result)


def plot_pemirl_smoke_results(
    pemirl_closenesses,
    filename=str(PLOTS_DIR / "pemirl_smoke_plot.pdf"),
    txt_filename=str(TABLES_DIR / "pemirl_smoke_plot.txt"),
):
    ensure_output_dirs()
    values = np.array(pemirl_closenesses).reshape(-1)
    mean_val = float(np.mean(values))
    if values.shape[0] > 1:
        se_val = float(np.std(values, ddof=1) / np.sqrt(values.shape[0]))
    else:
        se_val = 0.0

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(6, 4), layout="constrained")
    plt.bar(
        ["PEMIRL Smoke"],
        [mean_val],
        yerr=[se_val],
        capsize=8,
        color=sns.color_palette("muted")[7],
        alpha=0.85,
    )
    plt.ylabel("Average Scaled Goal Proximity", fontsize=14)
    plt.ylim(bottom=min(0.0, mean_val - se_val - 0.05))
    plt.title("PEMIRL End-to-End Smoke", fontsize=15)
    plt.savefig(str(filename), format="pdf", bbox_inches="tight")
    plt.close()

    with open(str(txt_filename), "w") as f:
        f.write(f"num_runs={values.shape[0]}\n")
        f.write(f"mean={mean_val:.6f}\n")
        f.write(f"stderr={se_val:.6f}\n")
        f.write(f"values={values.tolist()}\n")


def process_pemirl_smoke(num_runs=1, use_cache=False):
    print("Starting PEMIRL smoke processing...")
    ensure_output_dirs()
    cache_path = CACHE_DIR / "pemirl_smoke_results.npz"
    if use_cache and os.path.exists(cache_path):
        print("Loading cached PEMIRL smoke results...")
        results = np.load(cache_path)
        pemirl_closenesses = results["pemirl_closenesses"]
        results.close()
    else:
        pemirl_closenesses = get_pemirl_smoke_results(num_runs=num_runs)
        print("Saving PEMIRL smoke results to cache...")
        np.savez(cache_path, pemirl_closenesses=pemirl_closenesses)

    plot_pemirl_smoke_results(
        pemirl_closenesses,
        filename=str(PLOTS_DIR / "pemirl_smoke_plot.pdf"),
        txt_filename=str(TABLES_DIR / "pemirl_smoke_plot.txt"),
    )
    print("PEMIRL smoke processing complete.")


def get_noise_results(sgi_sai=False):
    # first process the RL runs
    model_idxs = list(range(num_runs))
    noise_coeffs = [0.0, 0.35, 0.60, 0.76, 0.87, 0.95, 1.0]
    print(f"Noise coefficients: {noise_coeffs}")

    rl_general_config = "general/default_experiment.yml"
    rl_train_config = "train/skip.yml"
    rl_dataset_config = "datasets/reach_noise.yml"
    rl_obs_dataset_config = "datasets/reach_for_pickplace.yml"
    rl_model_config = "model/load_default.yml"
    rl_rl_config = "rl/tqc_quarter_legacy_paper_esr_5m.yml"
    rl_inference_config = "inference/default_only_rl.yml"
    rl_baseline_config = "baselines/default.yml"

    print("Loading RL configurations...")
    rl_general_config = load_config_with_defaults(rl_general_config)
    rl_dataset_config = load_config_with_defaults(rl_dataset_config)
    rl_obs_dataset_config = load_config_with_defaults(rl_obs_dataset_config)
    rl_model_config = load_config_with_defaults(rl_model_config)
    rl_train_config = load_config_with_defaults(rl_train_config)
    rl_rl_config = load_config_with_defaults(rl_rl_config)
    rl_inference_config = load_config_with_defaults(rl_inference_config)
    rl_baseline_config = load_config_with_defaults(rl_baseline_config)

    print("Processing RL configurations...")
    process_configs(
        rl_general_config,
        rl_dataset_config,
        rl_obs_dataset_config,
        rl_model_config,
        rl_train_config,
        rl_rl_config,
        rl_inference_config,
        rl_baseline_config,
    )

    # now imitation
    im_general_config = "general/default_baselines.yml"
    im_train_config = "train/baselines.yml"
    im_dataset_config = "datasets/reach_noise.yml"
    im_obs_dataset_config = "datasets/reach_noise.yml"  # doesn't really matter, just need for process_configs()
    im_orl_dataset_config = "datasets/reach_noise.yml"
    im_model_config = "model/load_default.yml"
    im_rl_config = "rl/tqc_quarter_legacy_paper_esr_5m.yml"
    im_inference_config = "inference/reach_baselines.yml"
    im_baseline_configs = [
        "baselines/gail.yml",
        "baselines/airl.yml",
        "baselines/bc.yml",
    ]
    im_baseline_configs = _maybe_include_pemirl_baseline(im_baseline_configs)

    print("Loading imitation configurations...")
    im_general_config = load_config_with_defaults(im_general_config)
    im_dataset_config = load_config_with_defaults(im_dataset_config)
    im_obs_dataset_config = load_config_with_defaults(im_obs_dataset_config)
    im_orl_dataset_config = load_config_with_defaults(im_orl_dataset_config)
    im_model_config = load_config_with_defaults(im_model_config)
    im_train_config = load_config_with_defaults(im_train_config)
    im_rl_config = load_config_with_defaults(im_rl_config)
    im_inference_config = load_config_with_defaults(im_inference_config)
    im_baseline_configs = [
        load_config_with_defaults(im_baseline_config)
        for im_baseline_config in im_baseline_configs
    ]

    print("Processing imitation configurations...")
    for im_baseline_config in im_baseline_configs:
        process_configs(
            im_general_config,
            im_dataset_config,
            im_obs_dataset_config,
            im_model_config,
            im_train_config,
            im_rl_config,
            im_inference_config,
            im_baseline_config,
            orl_dataset_config=im_orl_dataset_config,
        )
    if "reach" in im_dataset_config.env:
        im_dataset_config.env = "reach-v2-goal-observable"
        im_train_config.env = "reach-v2-goal-observable"
        im_inference_config.env = "reach-v2-goal-observable"
    else:
        im_dataset_config.env = "pick-place-v2-goal-observable"
        im_train_config.env = "pick-place-v2-goal-observable"
        im_inference_config.env = "pick-place-v2-goal-observable"

    if sgi_sai:
        # python -m sri.run_experiment \
        # --general-config general/default_eval.yml \
        # --train-config train/skip.yml \
        # --dataset-config datasets/reach_noise.yml \
        # --obs-dataset-config datasets/reach_for_pickplace.yml \
        # --model-config model/goals.yml \
        # --rl-config rl/tqc_quarter_legacy_paper_esr_5m.yml \
        # --inference-config inference/eval_10_envs.yml \
        # --dataset-args "noise_coeff=$NOISE_COEFF" \
        # --model-idxs $MODEL_IDX \
        # --train-args "num_epochs=500" \
        # --model-args "load_model=True" \
        # --inference-args "batch_size=2, episodes=50" \
        # --rl-args "no_task_rep=False"
        sgi_general_config = "general/default_eval.yml"
        sgi_train_config = "train/skip.yml"
        sgi_dataset_config = "datasets/reach_noise.yml"
        sgi_obs_dataset_config = "datasets/reach_for_pickplace.yml"
        sgi_model_config = "model/goals.yml"
        sgi_rl_config = "rl/tqc_quarter_legacy_paper_esr_5m.yml"
        sgi_inference_config = "inference/eval_10_envs.yml"
        sgi_baseline_config = "baselines/default.yml"
        sgi_general_config = load_config_with_defaults(sgi_general_config)
        sgi_dataset_config = load_config_with_defaults(sgi_dataset_config)
        sgi_obs_dataset_config = load_config_with_defaults(sgi_obs_dataset_config)
        sgi_model_config = load_config_with_defaults(sgi_model_config)
        sgi_train_config = load_config_with_defaults(sgi_train_config)
        sgi_rl_config = load_config_with_defaults(sgi_rl_config)
        sgi_inference_config = load_config_with_defaults(sgi_inference_config)
        sgi_baseline_config = load_config_with_defaults(sgi_baseline_config)
        process_configs(
            sgi_general_config,
            sgi_dataset_config,
            sgi_obs_dataset_config,
            sgi_model_config,
            sgi_train_config,
            sgi_rl_config,
            sgi_inference_config,
            sgi_baseline_config,
        )
        # sgi_dataset_config.noise_coeff = noise_coeff
        # sgi_train_config.num_epochs = 2000
        sgi_train_config.num_epochs = 500
        sgi_model_config.load_model = True
        sgi_inference_config.batch_size = 2
        sgi_inference_config.episodes = 50
        sgi_rl_config.no_task_rep = False
        # maybe do the env setting thing here too

        # python -m sri.run_experiment \
        # --general-config general/default_eval.yml \
        # --train-config train/skip.yml \
        # --dataset-config datasets/reach_noise.yml \
        # --obs-dataset-config datasets/reach_for_pickplace.yml \
        # --model-config model/actions.yml \
        # --rl-config rl/tqc_quarter_legacy_paper_esr_5m.yml \
        # --inference-config inference/eval_10_envs.yml \
        # --dataset-args "noise_coeff=$NOISE_COEFF" \
        # --model-idxs $MODEL_IDX \
        # --train-args "num_epochs=2000" \
        # --model-args "load_model=True" \
        # --inference-args "batch_size=2, episodes=50, include_extra_reward_info=True" \
        # --rl-args "no_task_rep=False"
        sai_general_config = "general/default_eval.yml"
        sai_train_config = "train/skip.yml"
        sai_dataset_config = "datasets/reach_noise.yml"
        sai_obs_dataset_config = "datasets/reach_for_pickplace.yml"
        sai_model_config = "model/actions.yml"
        sai_rl_config = "rl/tqc_quarter_legacy_paper_esr_5m.yml"
        sai_inference_config = "inference/eval_10_envs.yml"
        sai_baseline_config = "baselines/default.yml"

        sai_general_config = load_config_with_defaults(sai_general_config)
        sai_dataset_config = load_config_with_defaults(sai_dataset_config)
        sai_obs_dataset_config = load_config_with_defaults(sai_obs_dataset_config)
        sai_model_config = load_config_with_defaults(sai_model_config)
        sai_train_config = load_config_with_defaults(sai_train_config)
        sai_rl_config = load_config_with_defaults(sai_rl_config)
        sai_inference_config = load_config_with_defaults(sai_inference_config)
        sai_baseline_config = load_config_with_defaults(sai_baseline_config)

        process_configs(
            sai_general_config,
            sai_dataset_config,
            sai_obs_dataset_config,
            sai_model_config,
            sai_train_config,
            sai_rl_config,
            sai_inference_config,
            sai_baseline_config,
        )
        # sgi_dataset and train config already set
        sai_train_config.num_epochs = 2000
        sai_model_config.load_model = True
        sai_inference_config.batch_size = 2
        sai_inference_config.episodes = 50
        sai_inference_config.include_extra_reward_info = True
        sai_rl_config.no_task_rep = False

        sgi_results = []
        sai_results = []

    print("Starting evaluation of runs...")
    rl_results = []
    gail_results = []
    airl_results = []
    bc_results = []
    pemirl_results = []
    for noise_coeff in tqdm(noise_coeffs, desc="Processing noise coefficients"):
        print(f"Processing noise coefficient: {noise_coeff}")
        rl_dataset_config.noise_coeff = noise_coeff
        im_dataset_config.noise_coeff = noise_coeff
        rl_results.append(
            get_eval_results(
                rl_general_config,
                rl_model_config,
                rl_train_config,
                rl_dataset_config,
                rl_obs_dataset_config,
                most_recent_first=True,
                model_idxs=model_idxs,
                rl=True,
                rl_config=rl_rl_config,
                baseline=False,
                inference_config=rl_inference_config,
            )
        )
        for im_baseline_config in im_baseline_configs:
            print(f"Processing baseline config: {im_baseline_config}")
            result = get_eval_results(
                im_general_config,
                im_model_config,
                im_train_config,
                im_dataset_config,
                im_obs_dataset_config,
                most_recent_first=True,
                model_idxs=model_idxs,
                rl=False,
                rl_config=im_rl_config,
                baseline=True,
                baselines_config=im_baseline_config,
                inference_config=im_inference_config,
            )
            _append_baseline_result(
                im_baseline_config,
                result,
                gail_results,
                airl_results,
                bc_results,
                pemirl_results,
            )

        if sgi_sai:
            sgi_dataset_config.noise_coeff = noise_coeff
            sai_dataset_config.noise_coeff = noise_coeff
            sgi_results.append(
                get_eval_results(
                    sgi_general_config,
                    sgi_model_config,
                    sgi_train_config,
                    sgi_dataset_config,
                    sgi_obs_dataset_config,
                    most_recent_first=True,
                    model_idxs=model_idxs,
                    rl=True,
                    # rl=False,
                    rl_config=sgi_rl_config,
                    baseline=False,
                    inference_config=sgi_inference_config,
                )
            )
            sai_results.append(
                get_eval_results(
                    sai_general_config,
                    sai_model_config,
                    sai_train_config,
                    sai_dataset_config,
                    sai_obs_dataset_config,
                    most_recent_first=True,
                    model_idxs=model_idxs,
                    rl=True,
                    # rl=False,
                    rl_config=sai_rl_config,
                    baseline=False,
                    inference_config=sai_inference_config,
                )
            )
    print("Evaluation complete.")

    rl_closenesses = np.array(rl_results)
    gail_closenesses = np.array(gail_results)
    airl_closenesses = np.array(airl_results)
    bc_closenesses = np.array(bc_results)
    print(
        f"Results shapes: RL: {rl_closenesses.shape}, GAIL: {gail_closenesses.shape}, AIRL: {airl_closenesses.shape}, BC: {bc_closenesses.shape}"
    )
    pemirl_closenesses = _optional_array(pemirl_results)
    if pemirl_closenesses is not None:
        print(f"Results shapes: PEMIRL: {pemirl_closenesses.shape}")
    if sgi_sai:
        # breakpoint()
        sgi_closenesses = np.array(sgi_results)
        sai_closenesses = np.array(sai_results)
        print(
            f"Results shapes: SGI: {sgi_closenesses.shape}, SAI: {sai_closenesses.shape}"
        )
    else:
        sgi_closenesses = None
        sai_closenesses = None
    return (
        noise_coeffs,
        rl_closenesses,
        gail_closenesses,
        airl_closenesses,
        bc_closenesses,
        pemirl_closenesses,
        sgi_closenesses,
        sai_closenesses,
    )

def process_noise(gt_baselines=1.0, num_runs=10, use_cache=False, sgi_sai=False):
    print("Starting process_noise function...")
    ensure_output_dirs()
    cache_path = CACHE_DIR / "noise_results.npz"
    if use_cache and os.path.exists(cache_path):
        print("Loading cached results...")
        results = np.load(cache_path)
        noise_coeffs = results["noise_coeffs"]
        rl_closenesses = results["rl_closenesses"]
        gail_closenesses = results["gail_closenesses"]
        airl_closenesses = results["airl_closenesses"]
        bc_closenesses = results["bc_closenesses"]
        pemirl_closenesses = _load_optional_array(results, "pemirl_closenesses")
        if sgi_sai:
            sgi_closenesses = results["sgi_closenesses"]
            sai_closenesses = results["sai_closenesses"]
        else:
            sgi_closenesses = None
            sai_closenesses = None
        results.close()
    else:
        (
            noise_coeffs,
            rl_closenesses,
            gail_closenesses,
            airl_closenesses,
            bc_closenesses,
            pemirl_closenesses,
            sgi_closenesses,
            sai_closenesses,
        ) = get_noise_results(sgi_sai=sgi_sai)
        # if use_cache:
        print("Saving results to cache...")
        np.savez(
            cache_path,
            noise_coeffs=noise_coeffs,
            rl_closenesses=rl_closenesses,
            gail_closenesses=gail_closenesses,
            airl_closenesses=airl_closenesses,
            bc_closenesses=bc_closenesses,
            pemirl_closenesses=_save_optional_array(pemirl_closenesses),
            sgi_closenesses=sgi_closenesses,
            sai_closenesses=sai_closenesses,
        )
    print("Plotting and saving results...")
    # plot_and_save_results(noise_coeffs, rl_closenesses, gail_closenesses, airl_closenesses, bc_closenesses, gt_baseline, filename="noise_plot.svg")
    plot_and_save_results_with_ci_stylish(
        noise_coeffs,
        rl_closenesses,
        sgi_closenesses,
        sai_closenesses,
        gail_closenesses,
        airl_closenesses,
        bc_closenesses,
        pemirl_closenesses,
        gt_baselines,
        filename=str(PLOTS_DIR / "noise_plot_ci.pdf"),
        n_bootstrap=1000,
        ci_percent=95,
        studentize=False,
        equal_spacing=True,
        x_label="Proportion of Random Actions ε",
        y_label="Average Scaled Goal Proximity",
    )
    print("Process complete.")

def get_adjust_results(sgi_sai=False):
    # Process the RL runs with goal position adjustments
    model_idxs = list(range(30))
    adjustments = [1.0, 0.6, 0.2, -0.2, -0.6, -1.0]
    print(f"Adjustments: {adjustments}")

    rl_general_config = "general/default_eval.yml"
    rl_train_config = "train/n1_skip.yml"
    rl_dataset_config = "datasets/reach_goal_pos_adjustment.yml"
    # rl_obs_dataset_config = "datasets/reach_noise.yml"
    # rl_orl_dataset_config = "datasets/reach_noise.yml"
    rl_obs_dataset_config = "datasets/reach_for_pickplace.yml"
    rl_model_config = "model/load_default.yml"
    rl_rl_config = "rl/tqc_quarter_legacy_paper_esr_5m.yml"
    rl_inference_config = "inference/eval_20_envs.yml"
    rl_baseline_config = "baselines/default.yml"

    print("Loading RL configurations...")
    rl_general_config = load_config_with_defaults(rl_general_config)
    rl_dataset_config = load_config_with_defaults(rl_dataset_config)
    rl_obs_dataset_config = load_config_with_defaults(rl_obs_dataset_config)

    rl_model_config = load_config_with_defaults(rl_model_config)
    rl_train_config = load_config_with_defaults(rl_train_config)
    rl_rl_config = load_config_with_defaults(rl_rl_config)
    rl_inference_config = load_config_with_defaults(rl_inference_config)
    rl_baseline_config = load_config_with_defaults(rl_baseline_config)

    print("Processing RL configurations...")
    process_configs(
        rl_general_config,
        rl_dataset_config,
        rl_obs_dataset_config,
        rl_model_config,
        rl_train_config,
        rl_rl_config,
        rl_inference_config,
        rl_baseline_config,
    )

    # Now imitation
    im_general_config = "general/default_baselines.yml"
    im_train_config = "train/n1_baselines.yml"
    im_dataset_config = "datasets/reach_goal_pos_adjustment.yml"
    im_obs_dataset_config = "datasets/reach_noise.yml"
    im_orl_dataset_config = "datasets/reach_noise.yml"
    im_model_config = "model/load_default.yml"
    im_rl_config = "rl/tqc_quarter_legacy_paper_esr_5m.yml"
    im_inference_config = "inference/reach_baselines_n1.yml"
    im_baseline_configs = [
        "baselines/gail.yml",
        "baselines/airl.yml",
        "baselines/bc.yml",
    ]
    im_baseline_configs = _maybe_include_pemirl_baseline(im_baseline_configs)

    print("Loading imitation configurations...")
    im_general_config = load_config_with_defaults(im_general_config)
    im_dataset_config = load_config_with_defaults(im_dataset_config)
    im_obs_dataset_config = load_config_with_defaults(im_obs_dataset_config)
    im_orl_dataset_config = load_config_with_defaults(im_orl_dataset_config)
    im_model_config = load_config_with_defaults(im_model_config)
    im_train_config = load_config_with_defaults(im_train_config)
    im_rl_config = load_config_with_defaults(im_rl_config)
    im_inference_config = load_config_with_defaults(im_inference_config)
    im_baseline_configs = [
        load_config_with_defaults(im_baseline_config)
        for im_baseline_config in im_baseline_configs
    ]

    print("Processing imitation configurations...")
    for im_baseline_config in im_baseline_configs:
        process_configs(
            im_general_config,
            im_dataset_config,
            im_obs_dataset_config,
            im_model_config,
            im_train_config,
            im_rl_config,
            im_inference_config,
            im_baseline_config,
            orl_dataset_config=im_orl_dataset_config,
        )
        
    if sgi_sai:
        # python -m sri.run_experiment \
        # --general-config general/default_eval.yml \
        # --train-config train/n1_skip.yml \
        # --dataset-config datasets/reach_goal_pos_adjustment.yml \
        # --obs-dataset-config datasets/reach_for_pickplace.yml \
        # --model-config model/goals.yml \
        # --rl-config rl/tqc_quarter_legacy_paper_esr_5m.yml \
        # --inference-config inference/eval_20_envs.yml \
        # --dataset-args "goal_pos_adjustment_factor=$ADJUSTMENT" \
        # --model-idxs $DATASET_IDX \
        # --train-args "num_epochs=500" \
        # --model-args "load_model=True" \
        # --inference-args "batch_size=2, episodes=50" \
        # --rl-args "no_task_rep=False"
        sgi_general_config = "general/default_eval.yml"
        sgi_train_config = "train/n1_skip.yml"
        sgi_dataset_config = "datasets/reach_goal_pos_adjustment.yml"
        sgi_obs_dataset_config = "datasets/reach_for_pickplace.yml"
        sgi_model_config = "model/goals.yml"
        sgi_rl_config = "rl/tqc_quarter_legacy_paper_esr_5m.yml"
        sgi_inference_config = "inference/eval_20_envs.yml"
        sgi_baseline_config = "baselines/default.yml"

        sgi_general_config = load_config_with_defaults(sgi_general_config)
        sgi_dataset_config = load_config_with_defaults(sgi_dataset_config)
        sgi_obs_dataset_config = load_config_with_defaults(sgi_obs_dataset_config)
        sgi_model_config = load_config_with_defaults(sgi_model_config)
        sgi_train_config = load_config_with_defaults(sgi_train_config)
        sgi_rl_config = load_config_with_defaults(sgi_rl_config)
        sgi_inference_config = load_config_with_defaults(sgi_inference_config)
        sgi_baseline_config = load_config_with_defaults(sgi_baseline_config)

        process_configs(
            sgi_general_config,
            sgi_dataset_config,
            sgi_obs_dataset_config,
            sgi_model_config,
            sgi_train_config,
            sgi_rl_config,
            sgi_inference_config,
            sgi_baseline_config,
        )

        sgi_train_config.num_epochs = 500
        sgi_model_config.load_model = True
        sgi_inference_config.batch_size = 2
        sgi_inference_config.episodes = 50
        sgi_rl_config.no_task_rep = False

        # python -m sri.run_experiment \
        # --general-config general/default_eval.yml \
        # --train-config train/n1_skip.yml \
        # --dataset-config datasets/reach_goal_pos_adjustment.yml \
        # --obs-dataset-config datasets/reach_for_pickplace.yml \
        # --model-config model/actions.yml \
        # --rl-config rl/tqc_quarter_legacy_paper_esr_5m.yml \
        # --inference-config inference/eval_20_envs.yml \
        # --dataset-args "goal_pos_adjustment_factor=$ADJUSTMENT" \
        # --model-idxs $DATASET_IDX \
        # --train-args "num_epochs=500" \
        # --model-args "load_model=True" \
        # --inference-args "batch_size=2, episodes=50, include_extra_reward_info=True" \
        # --rl-args "no_task_rep=False"
        sai_general_config = "general/default_eval.yml"
        sai_train_config = "train/n1_skip.yml"
        sai_dataset_config = "datasets/reach_goal_pos_adjustment.yml"
        sai_obs_dataset_config = "datasets/reach_for_pickplace.yml"
        sai_model_config = "model/actions.yml"
        sai_rl_config = "rl/tqc_quarter_legacy_paper_esr_5m.yml"
        sai_inference_config = "inference/eval_20_envs.yml"
        sai_baseline_config = "baselines/default.yml"

        sai_general_config = load_config_with_defaults(sai_general_config)
        sai_dataset_config = load_config_with_defaults(sai_dataset_config)
        sai_obs_dataset_config = load_config_with_defaults(sai_obs_dataset_config)
        sai_model_config = load_config_with_defaults(sai_model_config)
        sai_train_config = load_config_with_defaults(sai_train_config)
        sai_rl_config = load_config_with_defaults(sai_rl_config)
        sai_inference_config = load_config_with_defaults(sai_inference_config)
        sai_baseline_config = load_config_with_defaults(sai_baseline_config)

        process_configs(
            sai_general_config,
            sai_dataset_config,
            sai_obs_dataset_config,
            sai_model_config,
            sai_train_config,
            sai_rl_config,
            sai_inference_config,
            sai_baseline_config,
        )

        sai_train_config.num_epochs = 500
        sai_model_config.load_model = True
        sai_inference_config.batch_size = 2
        sai_inference_config.episodes = 50
        sai_inference_config.include_extra_reward_info = True
        sai_rl_config.no_task_rep = False

        sgi_results = []
        sai_results = []

    print("Starting evaluation of runs...")
    rl_results = []
    gail_results = []
    airl_results = []
    bc_results = []
    pemirl_results = []
    for adjustment in tqdm(adjustments, desc="Processing goal adjustments"):
        print(f"Processing adjustment: {adjustment}")
        rl_dataset_config.goal_pos_adjustment_factor = adjustment
        im_dataset_config.goal_pos_adjustment_factor = adjustment
        if sgi_sai:
            sgi_dataset_config.goal_pos_adjustment_factor = adjustment
            sai_dataset_config.goal_pos_adjustment_factor = adjustment
        # ipdb.set_trace()
        rl_results.append(
            get_eval_results(
                rl_general_config,
                rl_model_config,
                rl_train_config,
                rl_dataset_config,
                rl_obs_dataset_config,
                most_recent_first=True,
                model_idxs=model_idxs,
                rl=True,
                rl_config=rl_rl_config,
                baseline=False,
                inference_config=rl_inference_config,
            )
        )
        for im_baseline_config in im_baseline_configs:
            print(f"Processing baseline config: {im_baseline_config}")
            result = get_eval_results(
                im_general_config,
                im_model_config,
                im_train_config,
                im_dataset_config,
                im_obs_dataset_config,
                most_recent_first=True,
                model_idxs=model_idxs,
                rl=False,
                rl_config=im_rl_config,
                baseline=True,
                baselines_config=im_baseline_config,
                inference_config=im_inference_config,
            )
            _append_baseline_result(
                im_baseline_config,
                result,
                gail_results,
                airl_results,
                bc_results,
                pemirl_results,
            )

        if sgi_sai:
            sgi_results.append(
                get_eval_results(
                    sgi_general_config,
                    sgi_model_config,
                    sgi_train_config,
                    sgi_dataset_config,
                    sgi_obs_dataset_config,
                    most_recent_first=True,
                    model_idxs=model_idxs,
                    # rl=True,
                    rl=False,
                    rl_config=sgi_rl_config,
                    baseline=False,
                    inference_config=sgi_inference_config,
                )
            )
            sai_results.append(
                get_eval_results(
                    sai_general_config,
                    sai_model_config,
                    sai_train_config,
                    sai_dataset_config,
                    sai_obs_dataset_config,
                    most_recent_first=True,
                    model_idxs=model_idxs,
                    # rl=True,
                    rl=False,
                    rl_config=sai_rl_config,
                    baseline=False,
                    inference_config=sai_inference_config,
                )
            )

    print("Evaluation complete.")

    rl_closenesses = np.array(rl_results)
    gail_closenesses = np.array(gail_results)
    airl_closenesses = np.array(airl_results)
    bc_closenesses = np.array(bc_results)
    print(
        f"Results shapes: RL: {rl_closenesses.shape}, GAIL: {gail_closenesses.shape}, AIRL: {airl_closenesses.shape}, BC: {bc_closenesses.shape}"
    )
    pemirl_closenesses = _optional_array(pemirl_results)
    if pemirl_closenesses is not None:
        print(f"Results shapes: PEMIRL: {pemirl_closenesses.shape}")
    if sgi_sai:
        sgi_closenesses = np.array(sgi_results)
        sai_closenesses = np.array(sai_results)
        print(
            f"Results shapes: SGI: {sgi_closenesses.shape}, SAI: {sai_closenesses.shape}"
        )
    else:
        sgi_closenesses = None
        sai_closenesses = None
    return (
        adjustments,
        rl_closenesses,
        gail_closenesses,
        airl_closenesses,
        bc_closenesses,
        pemirl_closenesses,
        sgi_closenesses,
        sai_closenesses,
    )

def process_adjust(gt_baselines=1.0, num_runs=10, use_cache=False, sgi_sai=False):
    print("Starting process_adjust function...")

    ensure_output_dirs()
    cache_path = CACHE_DIR / "adjust_results.npz"
    if use_cache and os.path.exists(cache_path):
        print("Loading cached results...")
        results = np.load(cache_path)
        adjustments = results["adjustments"]
        rl_closenesses = results["rl_closenesses"]
        gail_closenesses = results["gail_closenesses"]
        airl_closenesses = results["airl_closenesses"]
        bc_closenesses = results["bc_closenesses"]
        pemirl_closenesses = _load_optional_array(results, "pemirl_closenesses")
        if sgi_sai:
            sgi_closenesses = results["sgi_closenesses"]
            sai_closenesses = results["sai_closenesses"]
        else:
            sgi_closenesses = None
            sai_closenesses = None
        results.close()
    else:
        (
            adjustments,
            rl_closenesses,
            gail_closenesses,
            airl_closenesses,
            bc_closenesses,
            pemirl_closenesses,
            sgi_closenesses,
            sai_closenesses,
        ) = get_adjust_results(sgi_sai=sgi_sai)
        # if use_cache:
        print("Saving results to cache...")
        np.savez(
            cache_path,
            adjustments=adjustments,
            rl_closenesses=rl_closenesses,
            gail_closenesses=gail_closenesses,
            airl_closenesses=airl_closenesses,
            bc_closenesses=bc_closenesses,
            pemirl_closenesses=_save_optional_array(pemirl_closenesses),
            sgi_closenesses=sgi_closenesses,
            sai_closenesses=sai_closenesses,
        )

    print("Plotting and saving results...")
    # plot_and_save_results(adjustments, rl_closenesses, gail_closenesses, airl_closenesses, bc_closenesses, gt_baseline, filename="adjustment_plot.svg")
    plot_and_save_results_with_ci_stylish(
        adjustments,
        rl_closenesses,
        sgi_closenesses,
        sai_closenesses,
        gail_closenesses,
        airl_closenesses,
        bc_closenesses,
        pemirl_closenesses,
        gt_baselines,
        filename=str(PLOTS_DIR / "adjustment_plot_ci.pdf"),
        n_bootstrap=1000,
        ci_percent=95,
        studentize=False,
        reverse_x=True,
        x_label="Goal Position Offset Factor α",
        y_label="Average Scaled Goal Proximity",
    )
    print("Process complete.")

def get_n_results(sgi_sai=False):
    # Process the RL runs with different train configurations (n1, n10)
    model_idxs = list(range(30))
    # noise_coeff = 0.35
    noise_coeff = 0.87
    train_configs = ["train/n1_skip.yml", "train/n10_skip.yml", "train/skip.yml"]
    print(f"Train configurations: {train_configs}")

    rl_general_config = "general/default_experiment.yml"
    # rl_dataset_config = "datasets/reach_noise.yml"
    rl_dataset_config = "datasets/reach_noise_fixed.yml"
    # rl_obs_dataset_config = "datasets/reach_noise.yml"
    rl_obs_dataset_config = "datasets/reach_for_pickplace.yml"
    rl_model_config = "model/load_default.yml"
    rl_rl_config = "rl/tqc_quarter_legacy_paper_esr_5m.yml"
    rl_inference_config = "inference/eval_10_envs.yml"
    rl_baseline_config = "baselines/default.yml"

    print("Loading RL configurations...")
    rl_general_config = load_config_with_defaults(rl_general_config)
    rl_dataset_config = load_config_with_defaults(rl_dataset_config)
    rl_obs_dataset_config = load_config_with_defaults(rl_obs_dataset_config)
    rl_model_config = load_config_with_defaults(rl_model_config)
    rl_rl_config = load_config_with_defaults(rl_rl_config)
    rl_inference_config = load_config_with_defaults(rl_inference_config)
    rl_baseline_config = load_config_with_defaults(rl_baseline_config)
    rl_dataset_config.noise_coeff = noise_coeff

    print("Processing RL configurations...")
    print(rl_dataset_config.horizon)
    rl_results = []
    for train_config in tqdm(train_configs, desc="Processing train configurations"):
        print(f"Processing train config: {train_config}")
        train_config = load_config_with_defaults(train_config)
        process_configs(
            rl_general_config,
            rl_dataset_config,
            rl_obs_dataset_config,
            rl_model_config,
            train_config,
            rl_rl_config,
            rl_inference_config,
            rl_baseline_config,
        )
        rl_results.append(
            get_eval_results(
                rl_general_config,
                rl_model_config,
                train_config,
                rl_dataset_config,
                rl_obs_dataset_config,
                most_recent_first=True,
                model_idxs=model_idxs,
                rl=True,
                rl_config=rl_rl_config,
                baseline=False,
                inference_config=rl_inference_config,
            )
        )

    # Now imitation with different baseline and inference configs
    im_general_config = "general/default_baselines.yml"
    im_train_config = "train/baselines.yml"
    im_dataset_config = "datasets/reach_noise_fixed.yml"
    im_obs_dataset_config = "datasets/reach_noise.yml"
    im_orl_dataset_config = "datasets/reach_noise.yml"
    im_model_config = "model/load_default.yml"
    im_rl_config = "rl/tqc_quarter_legacy_paper_esr_5m.yml"
    im_baseline_configs = [
        "baselines/gail.yml",
        "baselines/airl.yml",
        "baselines/bc.yml",
    ]
    im_baseline_configs = _maybe_include_pemirl_baseline(im_baseline_configs)
    im_inference_configs = [
        "inference/reach_baselines_n1.yml",
        "inference/reach_baselines_n10.yml",
        "inference/reach_baselines.yml",
    ]

    print("Loading imitation configurations...")
    im_general_config = load_config_with_defaults(im_general_config)
    im_dataset_config = load_config_with_defaults(im_dataset_config)
    im_obs_dataset_config = load_config_with_defaults(im_obs_dataset_config)
    im_orl_dataset_config = load_config_with_defaults(im_orl_dataset_config)
    im_model_config = load_config_with_defaults(im_model_config)
    im_train_config = load_config_with_defaults(im_train_config)
    im_rl_config = load_config_with_defaults(im_rl_config)
    im_baseline_configs = [
        load_config_with_defaults(im_baseline_config)
        for im_baseline_config in im_baseline_configs
    ]
    im_inference_configs = [
        load_config_with_defaults(im_inference_config)
        for im_inference_config in im_inference_configs
    ]

    # add noise coeff
    im_dataset_config.noise_coeff = noise_coeff

    print("Processing imitation configurations...")
    gail_results = []
    airl_results = []
    bc_results = []
    pemirl_results = []
    # for baseline_config, inference_config in tqdm(zip(im_baseline_configs, im_inference_configs), desc="Processing baseline and inference configurations"):
    for baseline_config in im_baseline_configs:
        for inference_config in im_inference_configs:
            # print(f"Processing baseline config: {baseline_config}, inference config: {inference_config}")
            print(
                f"Processing baseline {baseline_config.baselines[0]} with n {inference_config.n}"
            )
            process_configs(
                im_general_config,
                im_dataset_config,
                im_obs_dataset_config,
                im_model_config,
                im_train_config,
                im_rl_config,
                inference_config,
                baseline_config,
                orl_dataset_config=im_orl_dataset_config,
            )

            result = get_eval_results(
                im_general_config,
                im_model_config,
                im_train_config,
                im_dataset_config,
                im_obs_dataset_config,
                most_recent_first=True,
                model_idxs=model_idxs,
                rl=False,
                rl_config=im_rl_config,
                baseline=True,
                baselines_config=baseline_config,
                inference_config=inference_config,
            )
            _append_baseline_result(
                baseline_config,
                result,
                gail_results,
                airl_results,
                bc_results,
                pemirl_results,
            )

    if sgi_sai:
        # python -m sri.run_experiment \
        # --general-config general/default_eval.yml \
        # --train-config $TRAIN_CONFIG \
        # --dataset-config datasets/reach_noise_fixed.yml \
        # --obs-dataset-config datasets/reach_for_pickplace.yml \
        # --model-config model/goals.yml \
        # --rl-config rl/tqc_quarter_legacy_paper_esr_5m.yml \
        # --inference-config inference/eval_20_envs.yml \
        # --dataset-args "noise_coeff=$NOISE_COEFF" \
        # --model-idxs $MODEL_IDX \
        # --train-args "num_epochs=500" \
        # --model-args "load_model=True" \
        # --inference-args "batch_size=2, episodes=50" \
        # --rl-args "no_task_rep=False"
        sgi_general_config = "general/default_eval.yml"
        sgi_dataset_config = "datasets/reach_noise_fixed.yml"
        sgi_obs_dataset_config = "datasets/reach_for_pickplace.yml"
        sgi_model_config = "model/goals.yml"
        sgi_rl_config = "rl/tqc_quarter_legacy_paper_esr_5m.yml"
        sgi_inference_config = "inference/eval_20_envs.yml"
        sgi_baseline_config = "baselines/default.yml"
        sgi_general_config = load_config_with_defaults(sgi_general_config)
        sgi_dataset_config = load_config_with_defaults(sgi_dataset_config)
        sgi_obs_dataset_config = load_config_with_defaults(sgi_obs_dataset_config)
        sgi_model_config = load_config_with_defaults(sgi_model_config)
        sgi_rl_config = load_config_with_defaults(sgi_rl_config)
        sgi_inference_config = load_config_with_defaults(sgi_inference_config)
        sgi_baseline_config = load_config_with_defaults(sgi_baseline_config)
        # process_configs(
        #     sgi_general_config,
        #     sgi_dataset_config,
        #     sgi_obs_dataset_config,
        #     sgi_model_config,
        #     sgi_rl_config,
        #     sgi_inference_config,
        #     sgi_baseline_config,
        # )
        sgi_dataset_config.noise_coeff = noise_coeff
        sgi_model_config.load_model = True
        sgi_inference_config.batch_size = 2
        sgi_inference_config.episodes = 50
        sgi_rl_config.no_task_rep = False

        # python -m sri.run_experiment \
        # --general-config general/default_eval.yml \
        # --train-config $TRAIN_CONFIG \
        # --dataset-config datasets/reach_noise_fixed.yml \
        # --obs-dataset-config datasets/reach_for_pickplace.yml \
        # --model-config model/actions.yml \
        # --rl-config rl/tqc_quarter_legacy_paper_esr_5m.yml \
        # --inference-config inference/eval_20_envs.yml \
        # --dataset-args "noise_coeff=$NOISE_COEFF" \
        # --model-idxs $MODEL_IDX \
        # --train-args "num_epochs=500" \
        # --model-args "load_model=True" \
        # --inference-args "batch_size=2, episodes=50, include_extra_reward_info=True" \
        # --rl-args "no_task_rep=False"

        sai_general_config = "general/default_eval.yml"
        sai_dataset_config = "datasets/reach_noise_fixed.yml"
        sai_obs_dataset_config = "datasets/reach_for_pickplace.yml"
        sai_model_config = "model/actions.yml"
        sai_rl_config = "rl/tqc_quarter_legacy_paper_esr_5m.yml"
        sai_inference_config = "inference/eval_20_envs.yml"
        sai_baseline_config = "baselines/default.yml"
        sai_general_config = load_config_with_defaults(sai_general_config)
        sai_dataset_config = load_config_with_defaults(sai_dataset_config)
        sai_obs_dataset_config = load_config_with_defaults(sai_obs_dataset_config)
        sai_model_config = load_config_with_defaults(sai_model_config)
        sai_rl_config = load_config_with_defaults(sai_rl_config)
        sai_inference_config = load_config_with_defaults(sai_inference_config)
        sai_baseline_config = load_config_with_defaults(sai_baseline_config)
        sai_dataset_config.noise_coeff = noise_coeff
        sai_model_config.load_model = True
        sai_inference_config.batch_size = 2
        sai_inference_config.episodes = 50
        sai_inference_config.include_extra_reward_info = True
        sai_rl_config.no_task_rep = False

        sgi_results = []
        sai_results = []

        for train_config_path in tqdm(train_configs, desc="Processing train configurations"):
            print(f"Processing train config: {train_config_path}")
            train_config = load_config_with_defaults(train_config_path)
            process_configs(
                sgi_general_config,
                sgi_dataset_config,
                sgi_obs_dataset_config,
                sgi_model_config,
                train_config,
                sgi_rl_config,
                sgi_inference_config,
                sgi_baseline_config,
            )
            process_configs(
                sai_general_config,
                sai_dataset_config,
                sai_obs_dataset_config,
                sai_model_config,
                train_config,
                sai_rl_config,
                sai_inference_config,
                sai_baseline_config,
            )
            train_config.num_epochs = 500
            sgi_results.append(
                get_eval_results(
                    sgi_general_config,
                    sgi_model_config,
                    train_config,
                    sgi_dataset_config,
                    sgi_obs_dataset_config,
                    most_recent_first=True,
                    model_idxs=model_idxs,
                    rl=False,
                    rl_config=sgi_rl_config,
                    baseline=False,
                    inference_config=sgi_inference_config,
                )
            )
            # breakpoint()
            sai_results.append(
                get_eval_results(
                    sai_general_config,
                    sai_model_config,
                    train_config,
                    sai_dataset_config,
                    sai_obs_dataset_config,
                    most_recent_first=True,
                    model_idxs=model_idxs,
                    rl=False,
                    rl_config=sai_rl_config,
                    baseline=False,
                    inference_config=sai_inference_config,
                )
            )


    print("Evaluation complete.")

    rl_closenesses = np.array(rl_results)
    gail_closenesses = np.array(gail_results)
    airl_closenesses = np.array(airl_results)
    bc_closenesses = np.array(bc_results)
    print(
        f"Results shapes: RL: {rl_closenesses.shape}, GAIL: {gail_closenesses.shape}, AIRL: {airl_closenesses.shape}",
        f"BC: {bc_closenesses.shape}",
    )
    pemirl_closenesses = _optional_array(pemirl_results)
    if pemirl_closenesses is not None:
        print(f"Results shapes: PEMIRL: {pemirl_closenesses.shape}")
    if sgi_sai:
        sgi_closenesses = np.array(sgi_results)
        sai_closenesses = np.array(sai_results)
        print(
            f"Results shapes: SGI: {sgi_closenesses.shape}, SAI: {sai_closenesses.shape}"
        )
    else:
        sgi_closenesses = None
        sai_closenesses = None
    return (
        rl_closenesses,
        gail_closenesses,
        airl_closenesses,
        bc_closenesses,
        pemirl_closenesses,
        sgi_closenesses,
        sai_closenesses,
    )

def process_n(gt_baselines=1.0, num_runs=10, use_cache=False, sgi_sai=False):
    print("Starting process_n function...")

    ensure_output_dirs()
    cache_path = CACHE_DIR / "n_results.npz"
    if use_cache and os.path.exists(cache_path):
        print("Loading cached results...")
        results = np.load(cache_path)
        rl_closenesses = results["rl_closenesses"]
        gail_closenesses = results["gail_closenesses"]
        airl_closenesses = results["airl_closenesses"]
        bc_closenesses = results["bc_closenesses"]
        pemirl_closenesses = _load_optional_array(results, "pemirl_closenesses")
        if sgi_sai:
            sgi_closenesses = results["sgi_closenesses"]
            sai_closenesses = results["sai_closenesses"]
        else:
            sgi_closenesses = None
            sai_closenesses = None
        results.close()
    else:
        # rl_closenesses, gail_closenesses, airl_closenesses, bc_closenesses = get_n_results()
        (
            rl_closenesses,
            gail_closenesses,
            airl_closenesses,
            bc_closenesses,
            pemirl_closenesses,
            sgi_closenesses,
            sai_closenesses,
        ) = get_n_results(sgi_sai=sgi_sai)
        # if use_cache:
        print("Saving results to cache...")
        np.savez(
            cache_path,
            rl_closenesses=rl_closenesses,
            gail_closenesses=gail_closenesses,
            airl_closenesses=airl_closenesses,
            bc_closenesses=bc_closenesses,
            pemirl_closenesses=_save_optional_array(pemirl_closenesses),
            sgi_closenesses=sgi_closenesses,
            sai_closenesses=sai_closenesses,
        )

    print("Plotting and saving results...")
    plot_and_save_results_with_ci_stylish(
        [1, 10, 100],
        rl_closenesses,
        sgi_closenesses,
        sai_closenesses,
        gail_closenesses,
        airl_closenesses,
        bc_closenesses,
        pemirl_closenesses,
        gt_baselines,
        filename=str(PLOTS_DIR / "n_experiment_plot_ci.pdf"),
        n_bootstrap=1000,
        ci_percent=95,
        studentize=False,
        # equal_spacing=True,
        log_x=True,
        reverse_x=False,
        include_no_op_baseline=True,
        x_label="Number of Inference Demonstrations",
        y_label="Average Scaled Goal Proximity",
    )
    print("Process complete.")

def get_data_results(sgi_sai=False):
    model_idxs = list(range(30))
    train_configs = [
        "train/08_100_skip.yml",
        "train/08_1000_skip.yml",
        "train/08_10000_skip.yml",
        "train/02_100_skip.yml",
        "train/02_1000_skip.yml",
        "train/02_10000_skip.yml",
        "train/005_100_skip.yml",
        "train/005_1000_skip.yml",
        "train/005_10000_skip.yml",
    ]
    rl_general_config = "general/default_experiment.yml"
    rl_dataset_config = "datasets/reach_mirrored_circling.yml"
    rl_obs_dataset_config = "datasets/reach_for_pickplace.yml"
    rl_model_config = "model/load_default.yml"
    rl_rl_config = "rl/tqc_quarter_legacy_paper_esr_5m.yml"
    rl_inference_config = "inference/eval_10_envs.yml"
    rl_baseline_config = "baselines/default.yml"

    print("Loading RL configurations...")
    rl_general_config = load_config_with_defaults(rl_general_config)
    rl_dataset_config = load_config_with_defaults(rl_dataset_config)
    rl_obs_dataset_config = load_config_with_defaults(rl_obs_dataset_config)
    rl_model_config = load_config_with_defaults(rl_model_config)
    rl_rl_config = load_config_with_defaults(rl_rl_config)
    rl_inference_config = load_config_with_defaults(rl_inference_config)
    rl_baseline_config = load_config_with_defaults(rl_baseline_config)

    print("Processing RL configurations...")
    rl_results = []
    for train_config in tqdm(train_configs, desc="Processing train configurations"):
        print(f"Processing train config: {train_config}")
        train_config = load_config_with_defaults(train_config)
        process_configs(
            rl_general_config,
            rl_dataset_config,
            rl_obs_dataset_config,
            rl_model_config,
            train_config,
            rl_rl_config,
            rl_inference_config,
            rl_baseline_config,
        )

        rl_results.append(
            get_eval_results(
                rl_general_config,
                rl_model_config,
                train_config,
                rl_dataset_config,
                rl_obs_dataset_config,
                most_recent_first=True,
                model_idxs=model_idxs,
                rl=True,
                rl_config=rl_rl_config,
                baseline=False,
                inference_config=rl_inference_config,
            )
        )

    # Now imitation with different baseline configs (just one inference config this time)
    im_general_config = "general/default_baselines.yml"
    im_train_config = "train/baselines.yml"
    im_dataset_config = "datasets/reach_mirrored_circling.yml"
    im_obs_dataset_config = "datasets/reach_mirrored_circling.yml"
    im_orl_dataset_config = "datasets/reach_mirrored_circling.yml"
    im_model_config = "model/load_default.yml"
    im_rl_config = "rl/tqc_quarter_legacy_paper_esr_5m.yml"
    im_baseline_configs = [
        "baselines/airl.yml",
        "baselines/gail.yml",
        "baselines/bc.yml",
    ]
    im_baseline_configs = _maybe_include_pemirl_baseline(im_baseline_configs)
    im_inference_config = "inference/reach_baselines_n10_eval.yml"

    print("Loading imitation configurations...")
    im_general_config = load_config_with_defaults(im_general_config)
    im_dataset_config = load_config_with_defaults(im_dataset_config)
    im_obs_dataset_config = load_config_with_defaults(im_obs_dataset_config)
    im_orl_dataset_config = load_config_with_defaults(im_orl_dataset_config)
    im_model_config = load_config_with_defaults(im_model_config)
    im_train_config = load_config_with_defaults(im_train_config)
    im_rl_config = load_config_with_defaults(im_rl_config)
    im_baseline_configs = [
        load_config_with_defaults(im_baseline_config)
        for im_baseline_config in im_baseline_configs
    ]
    im_inference_config = load_config_with_defaults(im_inference_config)

    print("Processing imitation configurations...")
    gail_results = []
    airl_results = []
    bc_results = []
    pemirl_results = []
    for baseline_config in im_baseline_configs:
        print(f"Processing baseline config: {baseline_config}")
        process_configs(
            im_general_config,
            im_dataset_config,
            im_obs_dataset_config,
            im_model_config,
            im_train_config,
            im_rl_config,
            im_inference_config,
            baseline_config,
            orl_dataset_config=im_orl_dataset_config,
        )

        result = get_eval_results(
            im_general_config,
            im_model_config,
            im_train_config,
            im_dataset_config,
            im_obs_dataset_config,
            most_recent_first=True,
            model_idxs=model_idxs,
            rl=False,
            rl_config=im_rl_config,
            baseline=True,
            baselines_config=baseline_config,
            inference_config=im_inference_config,
        )
        _append_baseline_result(
            baseline_config,
            result,
            gail_results,
            airl_results,
            bc_results,
            pemirl_results,
        )

    # --- SGI/SAI logic ---
    if sgi_sai:
        # python -m sri.run_experiment \
        # --general-config general/default_eval.yml \
        # --train-config $TRAIN_CONFIG \
        # --dataset-config datasets/reach_mirrored_circling.yml \
        # --obs-dataset-config datasets/reach_for_pickplace.yml \
        # --model-config model/goals.yml \
        # --rl-config rl/tqc_quarter_legacy_paper_esr_5m.yml \
        # --inference-config inference/eval_10_envs.yml \
        # --model-idxs $DATASET_IDX \
        # --train-args "num_epochs=500" \
        # --model-args "load_model=True" \
        # --inference-args "batch_size=2, episodes=50" \
        # --rl-args "no_task_rep=False"
        sgi_general_config = "general/default_eval.yml"
        sgi_dataset_config = "datasets/reach_mirrored_circling.yml"
        sgi_obs_dataset_config = "datasets/reach_for_pickplace.yml"
        sgi_model_config = "model/goals.yml"
        sgi_rl_config = "rl/tqc_quarter_legacy_paper_esr_5m.yml"
        sgi_inference_config = "inference/eval_10_envs.yml"
        sgi_baseline_config = "baselines/default.yml"
        sgi_general_config = load_config_with_defaults(sgi_general_config)
        sgi_dataset_config = load_config_with_defaults(sgi_dataset_config)
        sgi_obs_dataset_config = load_config_with_defaults(sgi_obs_dataset_config)
        sgi_model_config = load_config_with_defaults(sgi_model_config)
        sgi_rl_config = load_config_with_defaults(sgi_rl_config)
        sgi_inference_config = load_config_with_defaults(sgi_inference_config)
        sgi_baseline_config = load_config_with_defaults(sgi_baseline_config)
        # process_configs(
        #     sgi_general_config,
        #     sgi_dataset_config,
        #     sgi_obs_dataset_config,
        #     sgi_model_config,
        #     sgi_rl_config,
        #     sgi_inference_config,
        #     sgi_baseline_config,
        # )
        sgi_model_config.load_model = True
        sgi_inference_config.batch_size = 2
        sgi_inference_config.episodes = 50
        sgi_rl_config.no_task_rep = False

        # python -m sri.run_experiment \
        # --general-config general/default_eval.yml \
        # --train-config $TRAIN_CONFIG \
        # --dataset-config datasets/reach_mirrored_circling.yml \
        # --obs-dataset-config datasets/reach_for_pickplace.yml \
        # --model-config model/actions.yml \
        # --rl-config rl/tqc_quarter_legacy_paper_esr_5m.yml \
        # --inference-config inference/eval_10_envs.yml \
        # --model-idxs $DATASET_IDX \
        # --train-args "num_epochs=500" \
        # --model-args "load_model=True" \
        # --inference-args "batch_size=2, episodes=50, include_extra_reward_info=True" \
        # --rl-args "no_task_rep=False"
        sai_general_config = "general/default_eval.yml"
        sai_dataset_config = "datasets/reach_mirrored_circling.yml"
        sai_obs_dataset_config = "datasets/reach_for_pickplace.yml"
        sai_model_config = "model/actions.yml"
        sai_rl_config = "rl/tqc_quarter_legacy_paper_esr_5m.yml"
        sai_inference_config = "inference/eval_10_envs.yml"
        sai_baseline_config = "baselines/default.yml"
        sai_general_config = load_config_with_defaults(sai_general_config)
        sai_dataset_config = load_config_with_defaults(sai_dataset_config)
        sai_obs_dataset_config = load_config_with_defaults(sai_obs_dataset_config)
        sai_model_config = load_config_with_defaults(sai_model_config)
        sai_rl_config = load_config_with_defaults(sai_rl_config)
        sai_inference_config = load_config_with_defaults(sai_inference_config)
        sai_baseline_config = load_config_with_defaults(sai_baseline_config)
        # process_configs(
        #     sai_general_config,
        #     sai_dataset_config,
        #     sai_obs_dataset_config,
        #     sai_model_config,
        #     sai_rl_config,
        #     sai_inference_config,
        #     sai_baseline_config,
        # )
        sai_model_config.load_model = True
        sai_inference_config.batch_size = 2
        sai_inference_config.episodes = 50
        sai_inference_config.include_extra_reward_info = True
        sai_rl_config.no_task_rep = False

        sgi_results = []
        sai_results = []

        for train_config in tqdm(train_configs, desc="Processing SGI/SAI train configs"):
            print(f"Processing SGI/SAI train config: {train_config}")
            train_config_loaded = load_config_with_defaults(train_config)
            train_config_loaded.num_epochs = 500
            process_configs(
                sgi_general_config,
                sgi_dataset_config,
                sgi_obs_dataset_config,
                sgi_model_config,
                train_config_loaded,
                sgi_rl_config,
                sgi_inference_config,
                sgi_baseline_config,
            )
            sgi_results.append(
                get_eval_results(
                    sgi_general_config,
                    sgi_model_config,
                    train_config_loaded,
                    sgi_dataset_config,
                    sgi_obs_dataset_config,
                    most_recent_first=True,
                    model_idxs=model_idxs,
                    rl=False,
                    rl_config=sgi_rl_config,
                    baseline=False,
                    inference_config=sgi_inference_config,
                )
            )
            process_configs(
                sai_general_config,
                sai_dataset_config,
                sai_obs_dataset_config,
                sai_model_config,
                train_config_loaded,
                sai_rl_config,
                sai_inference_config,
                sai_baseline_config,
            )
            sai_results.append(
                get_eval_results(
                    sai_general_config,
                    sai_model_config,
                    train_config_loaded,
                    sai_dataset_config,
                    sai_obs_dataset_config,
                    most_recent_first=True,
                    model_idxs=model_idxs,
                    rl=False,
                    rl_config=sai_rl_config,
                    baseline=False,
                    inference_config=sai_inference_config,
                )
            )
    else:
        sgi_results = None
        sai_results = None

    print("Evaluation complete.")

    rl_closenesses = np.array(rl_results)
    rl_closenesses = rl_closenesses.reshape((3, 3, len(model_idxs)))
    gail_closenesses = np.array(gail_results).squeeze()
    airl_closenesses = np.array(airl_results).squeeze()
    bc_closenesses = np.array(bc_results).squeeze()
    if sgi_sai:
        sgi_closenesses = np.array(sgi_results).squeeze()
        sai_closenesses = np.array(sai_results).squeeze()
        sgi_closenesses = sgi_closenesses.reshape((3, 3, len(model_idxs)))
        sai_closenesses = sai_closenesses.reshape((3, 3, len(model_idxs)))
        print(
            f"Results shapes: RL: {rl_closenesses.shape}, GAIL: {gail_closenesses.shape}, AIRL: {airl_closenesses.shape}, BC: {bc_closenesses.shape}, SGI: {sgi_closenesses.shape}, SAI: {sai_closenesses.shape}"
        )
    else:
        sgi_closenesses = None
        sai_closenesses = None
        print(
            f"Results shapes: RL: {rl_closenesses.shape}, GAIL: {gail_closenesses.shape}, AIRL: {airl_closenesses.shape}, BC: {bc_closenesses.shape}"
        )
    pemirl_closenesses = _optional_array(pemirl_results)
    if pemirl_closenesses is not None:
        print(f"Results shapes: PEMIRL: {pemirl_closenesses.shape}")
    return (
        rl_closenesses,
        gail_closenesses,
        airl_closenesses,
        bc_closenesses,
        pemirl_closenesses,
        sgi_closenesses,
        sai_closenesses,
    )

def process_data(gt_baselines=1.0, num_runs=10, use_cache=False, sgi_sai=False):
    print("Starting process_data function...")

    ensure_output_dirs()
    cache_path = CACHE_DIR / "data_results.npz"
    if use_cache and os.path.exists(cache_path):
        print("Loading cached results...")
        results = np.load(cache_path)
        rl_closenesses = results["rl_closenesses"]
        gail_closenesses = results["gail_closenesses"]
        airl_closenesses = results["airl_closenesses"]
        bc_closenesses = results["bc_closenesses"]
        pemirl_closenesses = _load_optional_array(results, "pemirl_closenesses")
        if sgi_sai:
            sgi_closenesses = results["sgi_closenesses"]
            sai_closenesses = results["sai_closenesses"]
        else:
            sgi_closenesses = None
            sai_closenesses = None
        results.close()
    else:
        (
            rl_closenesses,
            gail_closenesses,
            airl_closenesses,
            bc_closenesses,
            pemirl_closenesses,
            sgi_closenesses,
            sai_closenesses,
        ) = get_data_results(sgi_sai=sgi_sai)
        # if use_cache:
        print("Saving results to cache...")
        np.savez(
            cache_path,
            rl_closenesses=rl_closenesses,
            gail_closenesses=gail_closenesses,
            airl_closenesses=airl_closenesses,
            bc_closenesses=bc_closenesses,
            pemirl_closenesses=_save_optional_array(pemirl_closenesses),
            sgi_closenesses=sgi_closenesses,
            sai_closenesses=sai_closenesses,
        )

    print("Plotting and saving results...")
    y_values = 1600 * np.array([0.8, 0.2, 0.05])
    plot_and_save_results_with_ci_stylish(
        [100, 1000, 10000],
        rl_closenesses,
        sgi_closenesses,
        sai_closenesses,
        gail_closenesses,
        airl_closenesses,
        bc_closenesses,
        pemirl_closenesses,
        gt_baselines,
        filename=str(PLOTS_DIR / "data_experiment_plot_ci.pdf"),
        x_label="Number of Labeled Observations per Task",
        y_label="Number of Training Tasks",
        n_bootstrap=1000,
        ci_percent=95,
        studentize=False,
        y_values=y_values,
        txt_filename=str(TABLES_DIR / "data_experiment_plot_ci.txt"),
    )
    print("Process complete.")

def get_pickplace_results(sgi_sai=False):
    model_idxs = list(range(30))
    rl_general_config = "general/default_experiment.yml"
    rl_train_config = "train/pickplace_more_obs_005_02_skip.yml"
    rl_dataset_config = "datasets/reach_avoid_obj.yml"
    rl_obs_dataset_config = "datasets/default_pickplace.yml"
    rl_model_config = "model/load_default.yml"
    rl_rl_config = "rl/ppo_quarter_legacy_paper_esr_no_goal.yml"
    rl_inference_config = "inference/eval_20_envs_pp_single.yml"
    rl_baseline_config = "baselines/default.yml"

    print("Loading RL configurations...")
    rl_general_config = load_config_with_defaults(rl_general_config)
    rl_train_config = load_config_with_defaults(rl_train_config)
    rl_dataset_config = load_config_with_defaults(rl_dataset_config)
    rl_obs_dataset_config = load_config_with_defaults(rl_obs_dataset_config)
    rl_model_config = load_config_with_defaults(rl_model_config)
    rl_rl_config = load_config_with_defaults(rl_rl_config)
    rl_inference_config = load_config_with_defaults(rl_inference_config)
    rl_baseline_config = load_config_with_defaults(rl_baseline_config)

    print("Processing RL configurations...")
    process_configs(
        rl_general_config,
        rl_dataset_config,
        rl_obs_dataset_config,
        rl_model_config,
        rl_train_config,
        rl_rl_config,
        rl_inference_config,
        rl_baseline_config,
    )

    rl_result = get_eval_results(
        rl_general_config,
        rl_model_config,
        rl_train_config,
        rl_dataset_config,
        rl_obs_dataset_config,
        most_recent_first=True,
        model_idxs=model_idxs,
        rl=True,
        rl_config=rl_rl_config,
        baseline=False,
        inference_config=rl_inference_config,
    )

    # Now imitation
    im_general_config = "general/default_baselines.yml"
    im_train_config = "train/baselines.yml"
    im_dataset_config = "datasets/reach_avoid_obj.yml"
    im_obs_dataset_config = "datasets/default_pickplace.yml"
    im_model_config = "model/load_default.yml"
    im_rl_config = "rl/ppo_quarter_legacy_paper_esr_no_goal.yml"
    im_baseline_configs = [
        "baselines/gail_pickplace.yml",
        "baselines/airl_pickplace.yml",
        "baselines/bc_pickplace.yml",
    ]
    im_baseline_configs = _maybe_include_pemirl_baseline(im_baseline_configs)
    im_inference_config = "inference/pickplace_baselines_eval.yml"

    print("Loading imitation configurations...")
    im_general_config = load_config_with_defaults(im_general_config)
    im_dataset_config = load_config_with_defaults(im_dataset_config)
    im_obs_dataset_config = load_config_with_defaults(im_obs_dataset_config)
    im_model_config = load_config_with_defaults(im_model_config)
    im_train_config = load_config_with_defaults(im_train_config)
    im_rl_config = load_config_with_defaults(im_rl_config)
    im_baseline_configs = [
        load_config_with_defaults(cfg) for cfg in im_baseline_configs
    ]
    im_inference_config = load_config_with_defaults(im_inference_config)

    print("Processing imitation configurations...")
    gail_results, airl_results, bc_results, pemirl_results = [], [], [], []
    for im_cfg in im_baseline_configs:
        process_configs(
            im_general_config,
            im_dataset_config,
            im_obs_dataset_config,
            im_model_config,
            im_train_config,
            im_rl_config,
            im_inference_config,
            im_cfg,
        )
        res = get_eval_results(
            im_general_config,
            im_model_config,
            im_train_config,
            im_dataset_config,
            im_obs_dataset_config,
            most_recent_first=True,
            model_idxs=model_idxs,
            rl=False,
            rl_config=im_rl_config,
            baseline=True,
            baselines_config=im_cfg,
            inference_config=im_inference_config,
        )
        _append_baseline_result(
            im_cfg,
            res,
            gail_results,
            airl_results,
            bc_results,
            pemirl_results,
        )

    # SGI / SAI
    if sgi_sai:
        # SGI: goals
        # python -m sri.run_experiment \
        # --general-config general/default_eval.yml \
        # --train-config train/pickplace_more_obs_skip.yml \
        # --dataset-config datasets/reach_avoid_obj.yml \
        # --obs-dataset-config datasets/default_pickplace.yml \
        # --obs-dataset-config-2 datasets/reach_for_pickplace.yml \
        # --model-config model/goals.yml \
        # --rl-config rl/ppo_quarter_legacy_paper_esr_no_goal.yml \
        # --inference-config inference/eval_20_envs_pp_multitask.yml \
        # --model-idxs $MODEL_IDX \
        # --train-args "num_epochs=500" \
        # --model-args "load_model=True" \
        # --inference-args "batch_size=2, episodes=50" \
        # --rl-args "no_task_rep=False"
        sgi_general = load_config_with_defaults("general/default_eval.yml")
        sgi_train = load_config_with_defaults("train/pickplace_more_obs_skip.yml")
        sgi_dataset = load_config_with_defaults("datasets/reach_avoid_obj.yml")
        sgi_obs = load_config_with_defaults("datasets/default_pickplace.yml")
        sgi_model = load_config_with_defaults("model/goals.yml")
        sgi_rl = load_config_with_defaults("rl/ppo_quarter_legacy_paper_esr_no_goal.yml")
        sgi_infer = load_config_with_defaults("inference/eval_20_envs_pp_multitask.yml")
        sgi_base = load_config_with_defaults("baselines/default.yml")

        for cfg in (sgi_general, sgi_dataset, sgi_obs, sgi_model, sgi_rl, sgi_infer, sgi_base):
            cfg = cfg  # already loaded above

        process_configs(
            sgi_general,
            sgi_dataset,
            sgi_obs,
            sgi_model,
            sgi_train,
            sgi_rl,
            sgi_infer,
            sgi_base,
        )
        # override defaults
        sgi_train.num_epochs = 500
        sgi_model.load_model = True
        sgi_infer.batch_size = 2
        sgi_infer.episodes = 50
        sgi_rl.no_task_rep = False

        # python -m sri.run_experiment \
        # --general-config general/default_eval.yml \
        # --train-config train/pickplace_more_obs_skip.yml \
        # --dataset-config datasets/reach_avoid_obj.yml \
        # --obs-dataset-config datasets/default_pickplace.yml \
        # --obs-dataset-config-2 datasets/reach_for_pickplace.yml \
        # --model-config model/actions.yml \
        # --rl-config rl/ppo_quarter_legacy_paper_esr_no_goal.yml \
        # --inference-config inference/eval_20_envs_pp_multitask.yml \
        # --model-idxs $MODEL_IDX \
        # --train-args "num_epochs=500" \
        # --model-args "load_model=True" \
        # --inference-args "batch_size=2, episodes=50, include_extra_reward_info=True" \
        # --rl-args "no_task_rep=False"

        # SAI: actions + extra reward info
        sai_general = load_config_with_defaults("general/default_eval.yml")
        sai_train = load_config_with_defaults("train/pickplace_more_obs_skip.yml")
        # sai_dataset = rl_dataset_config
        sai_dataset = load_config_with_defaults("datasets/reach_avoid_obj.yml")
        # sai_obs = rl_obs_dataset_config
        sai_obs = load_config_with_defaults("datasets/default_pickplace.yml")
        sai_model = load_config_with_defaults("model/actions.yml")
        # sai_rl = rl_rl_config
        sai_rl = load_config_with_defaults("rl/ppo_quarter_legacy_paper_esr_no_goal.yml")
        sai_infer = load_config_with_defaults("inference/eval_20_envs_pp_multitask.yml")
        # sai_base = rl_baseline_config
        sai_base = load_config_with_defaults("baselines/default.yml")

        process_configs(
            sai_general,
            sai_dataset,
            sai_obs,
            sai_model,
            sai_train,
            sai_rl,
            sai_infer,
            sai_base,
        )
        sai_train.num_epochs = 500
        sai_model.load_model = True
        sai_infer.batch_size = 2
        sai_infer.episodes = 50
        sai_infer.include_extra_reward_info = True
        sai_rl.no_task_rep = False

        sgi_results = [get_eval_results(
            sgi_general,
            sgi_model,
            sgi_train,
            sgi_dataset,
            sgi_obs,
            most_recent_first=True,
            model_idxs=model_idxs,
            rl=True,
            rl_config=sgi_rl,
            baseline=False,
            inference_config=sgi_infer,
        )]
        sai_results = [get_eval_results(
            sai_general,
            sai_model,
            sai_train,
            sai_dataset,
            sai_obs,
            most_recent_first=True,
            model_idxs=model_idxs,
            rl=True,
            rl_config=sai_rl,
            baseline=False,
            inference_config=sai_infer,
        )]

        sgi_closenesses = np.array(sgi_results).squeeze()
        sai_closenesses = np.array(sai_results).squeeze()
    else:
        sgi_closenesses = None
        sai_closenesses = None

    rl_closenesses = np.array(rl_result).squeeze()
    gail_closenesses = np.array(gail_results).squeeze()
    airl_closenesses = np.array(airl_results).squeeze()
    bc_closenesses = np.array(bc_results).squeeze()
    print(
        f"Results shapes: RL: {rl_closenesses.shape}, GAIL: {gail_closenesses.shape}, AIRL: {airl_closenesses.shape}, BC: {bc_closenesses.shape}"
    )
    pemirl_closenesses = _optional_array(pemirl_results)
    if pemirl_closenesses is not None:
        print(f"Results shapes: PEMIRL: {pemirl_closenesses.shape}")
    if sgi_sai:
        print(
            f"Results shapes: SGI: {sgi_closenesses.shape}, SAI: {sai_closenesses.shape}"
        )

    return (
        rl_closenesses,
        gail_closenesses,
        airl_closenesses,
        bc_closenesses,
        pemirl_closenesses,
        sgi_closenesses,
        sai_closenesses,
    )


def process_pickplace(gt_baselines=1.0, num_runs=10, use_cache=False, sgi_sai=False):
    print("Starting process_pickplace function...")
    ensure_output_dirs()
    cache_path = CACHE_DIR / "pickplace_results.npz"
    if use_cache and os.path.exists(cache_path):
        print("Loading cached results...")
        results = np.load(cache_path)
        rl_closenesses = results["rl_closenesses"]
        gail_closenesses = results["gail_closenesses"]
        airl_closenesses = results["airl_closenesses"]
        bc_closenesses = results["bc_closenesses"]
        pemirl_closenesses = _load_optional_array(results, "pemirl_closenesses")
        if sgi_sai:
            sgi_closenesses = results["sgi_closenesses"]
            sai_closenesses = results["sai_closenesses"]
        else:
            sgi_closenesses = None
            sai_closenesses = None
        results.close()
    else:
        (
            rl_closenesses,
            gail_closenesses,
            airl_closenesses,
            bc_closenesses,
            pemirl_closenesses,
            sgi_closenesses,
            sai_closenesses,
        ) = get_pickplace_results(sgi_sai=sgi_sai)
        # if use_cache:
        print("Saving results to cache...")
        np.savez(
            cache_path,
            rl_closenesses=rl_closenesses,
            gail_closenesses=gail_closenesses,
            airl_closenesses=airl_closenesses,
            bc_closenesses=bc_closenesses,
            pemirl_closenesses=_save_optional_array(pemirl_closenesses),
            sgi_closenesses=sgi_closenesses,
            sai_closenesses=sai_closenesses,
        )

    print("Plotting and saving results...")
    plot_and_save_results_with_ci_stylish(
        None,
        rl_closenesses,
        sgi_closenesses,
        sai_closenesses,
        gail_closenesses,
        airl_closenesses,
        bc_closenesses,
        pemirl_closenesses,
        gt_baselines,
        filename=str(PLOTS_DIR / "pickplace_experiment_plot_ci.pdf"),
        n_bootstrap=1000,
        ci_percent=95,
        studentize=False,
        equal_spacing=True,
    )
    print("Process complete.")

def get_ppdata_results(sgi_sai=False):
    model_idxs = list(range(30))
    # NOISE_COEFFS=(0.0 0.35 0.60 0.76 0.87 0.95 1.0)
    # # NOISE_COEFF_IDX=$((IDX % ${#NOISE_COEFFS[@]}))
    # # TRAIN_CONFIGS=("train/08_100.yml" "train/08_1000.yml" "train/08_10000.yml" "train/02_100.yml" "train/02_1000.yml" "train/02_10000.yml" "train/005_100.yml" "train/005_1000.yml" "train/005_10000.yml")
    # # TRAIN_CONFIGS=("train/08_100_pp.yml" "train/08_1000_pp.yml" "train/08_10000_pp.yml" "train/02_100_pp.yml" "train/02_1000_pp.yml" "train/02_10000_pp.yml" "train/005_100_pp.yml" "train/005_1000_pp.yml" "train/005_10000_pp.yml")
    # TRAIN_CONFIGS=("train/08_100_pp_skip.yml" "train/08_1000_pp_skip.yml" "train/08_10000_pp_skip.yml" "train/02_100_pp_skip.yml" "train/02_1000_pp_skip.yml" "train/02_10000_pp_skip.yml" "train/005_100_pp_skip.yml" "train/005_1000_pp_skip.yml" "train/005_10000_pp_skip.yml")
    # TRAIN_CONFIG_IDX=$((IDX % ${#TRAIN_CONFIGS[@]}))
    # TRAIN_CONFIG=${TRAIN_CONFIGS[$TRAIN_CONFIG_IDX]}
    # DATASET_IDX=$((IDX / ${#TRAIN_CONFIGS[@]}))
    # MODEL_IDX=$DATASET_IDX

    # echo "NOISE_COEFF: $NOISE_COEFF"
    # echo "DATASET_IDX: $DATASET_IDX"
    # echo "TRAIN_CONFIG_IDX: $TRAIN_CONFIG_IDX"
    # echo "TRAIN_CONFIG: $TRAIN_CONFIG"

    # python -m sri.run_experiment_temp_2 \
    #     --general-config general/default_eval.yml \
    #     --train-config $TRAIN_CONFIG \
    #     --dataset-config datasets/reach_mirrored_circling_avoid_obj.yml \
    #     --obs-dataset-config datasets/default_pickplace.yml \
    #     --obs-dataset-config-2 datasets/reach_for_pickplace.yml \
    #     --model-config model/load_default.yml \
    #     --rl-config rl/ppo_quarter_legacy_paper_esr_no_goal.yml \
    #     --inference-config inference/eval_20_envs_pp_single.yml \
    #     --model-idxs $MODEL_IDX \
    train_configs = [
        "train/08_100_pp_skip.yml",
        "train/08_1000_pp_skip.yml",
        "train/08_10000_pp_skip.yml",
        "train/02_100_pp_skip.yml",
        "train/02_1000_pp_skip.yml",
        "train/02_10000_pp_skip.yml",
        "train/005_100_pp_skip.yml",
        "train/005_1000_pp_skip.yml",
        "train/005_10000_pp_skip.yml",
    ]
    # note that for simplicity we manually set start_prop_obs_1 and end_prop_obs_1 later
    rl_general_config = "general/default_experiment.yml"
    rl_dataset_config = "datasets/reach_mirrored_circling_avoid_obj.yml"
    rl_obs_dataset_config = "datasets/default_pickplace.yml"
    rl_orl_dataset_config = "datasets/reach_for_pickplace.yml"
    rl_model_config = "model/load_default.yml"
    rl_rl_config = "rl/ppo_quarter_legacy_paper_esr_no_goal.yml"
    rl_inference_config = "inference/eval_20_envs_pp_single.yml"
    rl_baseline_config = "baselines/default.yml"

    print("Loading RL configurations...")
    rl_general_config = load_config_with_defaults(rl_general_config)
    rl_dataset_config = load_config_with_defaults(rl_dataset_config)
    rl_obs_dataset_config = load_config_with_defaults(rl_obs_dataset_config)
    rl_model_config = load_config_with_defaults(rl_model_config)
    rl_rl_config = load_config_with_defaults(rl_rl_config)
    rl_inference_config = load_config_with_defaults(rl_inference_config)
    rl_baseline_config = load_config_with_defaults(rl_baseline_config)

    print("Processing RL configurations...")
    rl_results = []
    for train_config in tqdm(train_configs, desc="Processing train configurations"):
        print(f"Processing train config: {train_config}")
        train_config = load_config_with_defaults(train_config)
        train_config.start_prop_obs_1 = 0.05
        train_config.end_prop_obs_1 = 0.2
        process_configs(
            rl_general_config,
            rl_dataset_config,
            rl_obs_dataset_config,
            rl_model_config,
            train_config,
            rl_rl_config,
            rl_inference_config,
            rl_baseline_config,
        )
        rl_results.append(
            get_eval_results(
                rl_general_config,
                rl_model_config,
                train_config,
                rl_dataset_config,
                rl_obs_dataset_config,
                most_recent_first=True,
                model_idxs=model_idxs,
                rl=True,
                rl_config=rl_rl_config,
                baseline=False,
                inference_config=rl_inference_config,
            )
        )

    # Now imitation with different baseline configs (just one inference config this time)
    # BASELINE_CONFIGS=("baselines/airl.yml" "baselines/gail.yml" "baselines/bc.yml")
    # BASELINE_CONFIG_IDX=$((IDX % ${#BASELINE_CONFIGS[@]}))
    # BASELINE_CONFIG=${BASELINE_CONFIGS[$BASELINE_CONFIG_IDX]}
    # DATASET_IDX=$((IDX / ${#BASELINE_CONFIGS[@]}))
    # MODEL_IDX=$DATASET_IDX

    # echo "BASELINE_CONFIG_IDX: $BASELINE_CONFIG_IDX"
    # echo "BASELINE_CONFIG: $BASELINE_CONFIG"
    # echo "MODEL_IDX: $MODEL_IDX"

    # python -m sri.run_experiment_temp_2 \
    #     --general-config general/default_eval.yml \
    #     --train-config train/baselines.yml \
    #     --dataset-config datasets/reach_mirrored_circling.yml \
    #     --obs-dataset-config datasets/reach_mirrored_circling.yml \
    #     --orl-dataset-config datasets/reach_mirrored_circling.yml \
    #     --model-config model/load_default.yml \
    #     --rl-config rl/tqc_quarter_legacy_paper_esr_5m.yml \
    #     --inference-config inference/reach_baselines_n10_eval.yml \
    #     --baselines-config $BASELINE_CONFIG \
    #     --model-idx $MODEL_IDX \

    im_general_config = "general/default_baselines.yml"
    im_train_config = "train/n10_baselines.yml"
    im_dataset_config = "datasets/reach_mirrored_circling_avoid_obj.yml"
    im_obs_dataset_config = "datasets/default_pickplace.yml"
    im_model_config = "model/load_default.yml"
    im_rl_config = "rl/ppo_quarter_legacy_paper_esr_no_goal.yml"
    im_baseline_configs = [
        "baselines/airl_pickplace.yml",
        "baselines/gail_pickplace.yml",
        "baselines/bc_pickplace.yml",
    ]
    im_baseline_configs = _maybe_include_pemirl_baseline(im_baseline_configs)
    im_inference_config = "inference/pickplace_baselines_n10_eval.yml"

    print("Loading imitation configurations...")
    im_general_config = load_config_with_defaults(im_general_config)
    im_dataset_config = load_config_with_defaults(im_dataset_config)
    im_obs_dataset_config = load_config_with_defaults(im_obs_dataset_config)
    im_model_config = load_config_with_defaults(im_model_config)
    im_train_config = load_config_with_defaults(im_train_config)
    im_rl_config = load_config_with_defaults(im_rl_config)
    im_baseline_configs = [
        load_config_with_defaults(im_baseline_config)
        for im_baseline_config in im_baseline_configs
    ]
    im_inference_config = load_config_with_defaults(im_inference_config)

    print("Processing imitation configurations...")
    gail_results = []
    airl_results = []
    bc_results = []
    pemirl_results = []
    for baseline_config in im_baseline_configs:
        print(f"Processing baseline config: {baseline_config}")
        process_configs(
            im_general_config,
            im_dataset_config,
            im_obs_dataset_config,
            im_model_config,
            im_train_config,
            im_rl_config,
            im_inference_config,
            baseline_config,
        )

        result = get_eval_results(
            im_general_config,
            im_model_config,
            im_train_config,
            im_dataset_config,
            im_obs_dataset_config,
            most_recent_first=True,
            model_idxs=model_idxs,
            rl=False,
            rl_config=im_rl_config,
            baseline=True,
            baselines_config=baseline_config,
            inference_config=im_inference_config,
        )
        _append_baseline_result(
            baseline_config,
            result,
            gail_results,
            airl_results,
            bc_results,
            pemirl_results,
        )

    # SGI / SAI
    if sgi_sai:
        # python -m sri.run_experiment \
        # --general-config general/default_eval.yml \
        # --train-config train/n10_pp_skip.yml \
        # --dataset-config datasets/reach_mirrored_circling_avoid_obj.yml \
        # --obs-dataset-config datasets/default_pickplace.yml \
        # --obs-dataset-config-2 datasets/reach_for_pickplace.yml \
        # --model-config model/goals.yml \
        # --rl-config rl/ppo_quarter_legacy_paper_esr_no_goal.yml \
        # --inference-config inference/eval_20_envs_pp_multitask.yml \
        # --model-idxs $MODEL_IDX \
        # --train-args "train_split=$TRAIN_SPLIT, num_obs=$NUM_OBS, num_epochs=200" \
        # --model-args "load_model=True" \
        # --inference-args "batch_size=2, episodes=50" \
        # --rl-args "no_task_rep=False"
        sgi_general_config = "general/default_eval.yml"
        sgi_train_config = "train/n10_pp_skip.yml"
        sgi_dataset_config = "datasets/reach_mirrored_circling_avoid_obj.yml"
        sgi_obs_dataset_config = "datasets/default_pickplace.yml" 
        sgi_orl_dataset_config = "datasets/reach_for_pickplace.yml"
        sgi_model_config = "model/goals.yml"
        sgi_rl_config = "rl/ppo_quarter_legacy_paper_esr_no_goal.yml"
        sgi_inference_config = "inference/eval_20_envs_pp_multitask.yml"
        sgi_baseline_config = "baselines/default.yml"

        sgi_general_config = load_config_with_defaults(sgi_general_config)
        sgi_dataset_config = load_config_with_defaults(sgi_dataset_config)
        sgi_obs_dataset_config = load_config_with_defaults(sgi_obs_dataset_config)
        sgi_orl_dataset_config = load_config_with_defaults(sgi_orl_dataset_config)
        sgi_model_config = load_config_with_defaults(sgi_model_config)
        sgi_train_config = load_config_with_defaults(sgi_train_config)
        sgi_rl_config = load_config_with_defaults(sgi_rl_config)
        sgi_inference_config = load_config_with_defaults(sgi_inference_config)
        sgi_baseline_config = load_config_with_defaults(sgi_baseline_config)
        
        process_configs(
            sgi_general_config,
            sgi_dataset_config,
            sgi_obs_dataset_config,
            sgi_model_config,
            sgi_train_config,
            sgi_rl_config,
            sgi_inference_config,
            sgi_baseline_config,
        )
        
        sgi_train_config.num_epochs = 200
        sgi_model_config.load_model = True
        sgi_inference_config.batch_size = 2
        sgi_inference_config.episodes = 50
        sgi_rl_config.no_task_rep = False
        
        # python -m sri.run_experiment \
        # --general-config general/default_eval.yml \
        # --train-config train/n10_pp_skip.yml \
        # --dataset-config datasets/reach_mirrored_circling_avoid_obj.yml \
        # --obs-dataset-config datasets/default_pickplace.yml \
        # --obs-dataset-config-2 datasets/reach_for_pickplace.yml \
        # --model-config model/actions.yml \
        # --rl-config rl/ppo_quarter_legacy_paper_esr_no_goal.yml \
        # --inference-config inference/eval_20_envs_pp_multitask.yml \
        # --model-idxs $MODEL_IDX \
        # --train-args "train_split=$TRAIN_SPLIT, num_obs=$NUM_OBS, num_epochs=500" \
        # --model-args "load_model=True" \
        # --inference-args "batch_size=2, episodes=50, include_extra_reward_info=True" \
        # --rl-args "no_task_rep=False"
        sai_general_config = "general/default_eval.yml"
        sai_train_config = "train/n10_pp_skip.yml" 
        sai_dataset_config = "datasets/reach_mirrored_circling_avoid_obj.yml"
        sai_obs_dataset_config = "datasets/default_pickplace.yml"
        sai_orl_dataset_config = "datasets/reach_for_pickplace.yml"
        sai_model_config = "model/actions.yml"
        sai_rl_config = "rl/ppo_quarter_legacy_paper_esr_no_goal.yml"
        sai_inference_config = "inference/eval_20_envs_pp_multitask.yml"
        sai_baseline_config = "baselines/default.yml"
        
        sai_general_config = load_config_with_defaults(sai_general_config)
        sai_dataset_config = load_config_with_defaults(sai_dataset_config)
        sai_obs_dataset_config = load_config_with_defaults(sai_obs_dataset_config)
        sai_orl_dataset_config = load_config_with_defaults(sai_orl_dataset_config)
        sai_model_config = load_config_with_defaults(sai_model_config)
        sai_train_config = load_config_with_defaults(sai_train_config)
        sai_rl_config = load_config_with_defaults(sai_rl_config)
        sai_inference_config = load_config_with_defaults(sai_inference_config)
        sai_baseline_config = load_config_with_defaults(sai_baseline_config)
        
        process_configs(
            sai_general_config,
            sai_dataset_config,
            sai_obs_dataset_config,
            sai_model_config,
            sai_train_config,
            sai_rl_config,
            sai_inference_config,
            sai_baseline_config,
        )
        
        sai_train_config.num_epochs = 500
        sai_model_config.load_model = True
        sai_inference_config.batch_size = 2
        sai_inference_config.episodes = 50
        sai_inference_config.include_extra_reward_info = True
        sai_rl_config.no_task_rep = False

        sgi_results = []
        sai_results = []
        
        for train_config in tqdm(train_configs, desc="Processing SGI/SAI train configs"):
            print(f"Processing SGI/SAI train config: {train_config}")
            train_config_loaded = load_config_with_defaults(train_config)
            # Process SGI
            process_configs(
                sgi_general_config,
                sgi_dataset_config,
                sgi_obs_dataset_config,
                sgi_model_config,
                train_config_loaded,
                sgi_rl_config,
                sgi_inference_config,
                sgi_baseline_config,
            )
            train_config_loaded.num_epochs = 200
            sgi_results.append(
                get_eval_results(
                    sgi_general_config,
                    sgi_model_config,
                    train_config_loaded,
                    sgi_dataset_config,
                    sgi_obs_dataset_config,
                    most_recent_first=True,
                    model_idxs=model_idxs,
                    rl=False,
                    rl_config=sgi_rl_config,
                    baseline=False,
                    inference_config=sgi_inference_config,
                )
            )
            
            train_config_loaded.num_epochs = 500
            # Process SAI
            process_configs(
                sai_general_config,
                sai_dataset_config,
                sai_obs_dataset_config,
                sai_model_config,
                train_config_loaded,
                sai_rl_config,
                sai_inference_config,
                sai_baseline_config,
            )
            sai_results.append(
                get_eval_results(
                    sai_general_config,
                    sai_model_config,
                    train_config_loaded,
                    sai_dataset_config,
                    sai_obs_dataset_config,
                    most_recent_first=True,
                    model_idxs=model_idxs,
                    rl=False,
                    rl_config=sai_rl_config,
                    baseline=False,
                    inference_config=sai_inference_config,
                )
            )

    print("Evaluation complete.")

    rl_closenesses = np.array(rl_results)
    # reshape rl_closenesses
    # rl_closenesses = rl_closenesses.reshape((3, 3, 10))
    rl_closenesses = rl_closenesses.reshape((3, 3, len(model_idxs)))
    gail_closenesses = np.array(gail_results).squeeze()
    airl_closenesses = np.array(airl_results).squeeze()
    bc_closenesses = np.array(bc_results).squeeze()
    print(
        f"Results shapes: SRI: {rl_closenesses.shape}, GAIL: {gail_closenesses.shape}, AIRL: {airl_closenesses.shape}, BC: {bc_closenesses.shape}"
    )
    pemirl_closenesses = _optional_array(pemirl_results)
    if pemirl_closenesses is not None:
        print(f"Results shapes: PEMIRL: {pemirl_closenesses.shape}")
    
    if sgi_sai:
        sgi_closenesses = np.array(sgi_results).squeeze()
        sai_closenesses = np.array(sai_results).squeeze()
        sgi_closenesses = sgi_closenesses.reshape((3, 3, len(model_idxs)))
        sai_closenesses = sai_closenesses.reshape((3, 3, len(model_idxs)))
        print(f"Results shapes: SGI: {sgi_closenesses.shape}, SAI: {sai_closenesses.shape}")
        return (
            rl_closenesses,
            gail_closenesses,
            airl_closenesses,
            bc_closenesses,
            pemirl_closenesses,
            sgi_closenesses,
            sai_closenesses,
        )
    else:
        return (
            rl_closenesses,
            gail_closenesses,
            airl_closenesses,
            bc_closenesses,
            pemirl_closenesses,
            None,
            None,
        )


def process_ppdata(gt_baselines=1.0, num_runs=10, use_cache=False, sgi_sai=False):
    print("Starting process_data function...")

    ensure_output_dirs()
    cache_path = CACHE_DIR / "ppdata_results.npz"
    if use_cache and os.path.exists(cache_path):
        print("Loading cached results...")
        results = np.load(cache_path)
        rl_closenesses = results["rl_closenesses"]
        gail_closenesses = results["gail_closenesses"]
        airl_closenesses = results["airl_closenesses"]
        bc_closenesses = results["bc_closenesses"]
        pemirl_closenesses = _load_optional_array(results, "pemirl_closenesses")
        if sgi_sai:
            sgi_closenesses = results["sgi_closenesses"]
            sai_closenesses = results["sai_closenesses"]
        else:
            sgi_closenesses = None
            sai_closenesses = None
        results.close()
    else:
        (
            rl_closenesses,
            gail_closenesses,
            airl_closenesses,
            bc_closenesses,
            pemirl_closenesses,
            sgi_closenesses,
            sai_closenesses,
        ) = get_ppdata_results(sgi_sai=sgi_sai)
        # if use_cache:
        print("Saving results to cache...")
        np.savez(
            cache_path,
            rl_closenesses=rl_closenesses,
            gail_closenesses=gail_closenesses,
            airl_closenesses=airl_closenesses,
            bc_closenesses=bc_closenesses,
            pemirl_closenesses=_save_optional_array(pemirl_closenesses),
            sgi_closenesses=sgi_closenesses,
            sai_closenesses=sai_closenesses,
        )

    print("Plotting and saving results...")
    y_values = 1600 * np.array([0.8, 0.2, 0.05])
    plot_and_save_results_with_ci_stylish(
        [100, 1000, 10000],
        rl_closenesses,
        sgi_closenesses,
        sai_closenesses,
        gail_closenesses,
        airl_closenesses,
        bc_closenesses,
        pemirl_closenesses,
        gt_baselines,
        filename=str(PLOTS_DIR / "ppdata_experiment_plot_ci.pdf"),
        n_bootstrap=1000,
        ci_percent=95,
        studentize=False,
        y_values=y_values,
        txt_filename=str(TABLES_DIR / "ppdata_experiment_plot_ci.txt"),
        x_label="Number of Labeled Observations per Task",
        y_label="Number of Training Tasks",
    )
    print("Process complete.")

def standard_error(data_array, axis=0):
    # return np.std(data_array, ddof=1) / np.sqrt(len(data_array))
    return np.std(data_array, axis=axis, ddof=1) / np.sqrt(data_array.shape[axis])

def plot_and_save_results_with_ci_stylish(
    x_values,
    rl_closenesses,
    sgi_closenesses,
    sai_closenesses,
    gail_closenesses,
    airl_closenesses,
    bc_closenesses,
    pemirl_closenesses,
    gt_baselines,
    filename=str(PLOTS_DIR / "closeness_plot.pdf"),
    log_x=False,
    equal_spacing=False,
    n_bootstrap=1000,
    ci_percent=95,
    studentize=False,
    x_label="Noise Coefficient",
    y_label="Average Scaled Goal Proximity",
    y_values=None,  # New y_values kwarg to be used when rl_closenesses is square
    txt_filename=str(TABLES_DIR / "results.txt"),  # New argument for output text file
    reverse_x=False,
    ignore_gail=False,  # New argument to ignore GAIL
    clip_to_zero=True,  # New argument to clip individual trials to 0 if negative
    clip_avg_to_zero=False,  # New argument to clip average values to 0 if negative
    # y_axis_start_at_zero=False,  # New argument to set y-axis to start at 0
    y_axis_min=-0.05,
    include_no_op_baseline=True,  # New argument to include a no-op baseline (just always at 0.0)
    use_stderr=True,  # <--- NEW argument to use standard error instead of bootstrap
):
    """
    When use_stderr=True, we compute mean +/- SE instead of bootstrap CIs.
    In the text outputs, we display "mean ± se" instead of "mean (lower, upper)".
    In the plots, we use symmetrical error bars = ±SE around the mean.
    """

    ensure_output_dirs()
    filename = str(filename)
    txt_filename = str(txt_filename)

    if clip_to_zero:
        rl_closenesses = np.clip(rl_closenesses, 0, None)
        gail_closenesses = np.clip(gail_closenesses, 0, None)
        airl_closenesses = np.clip(airl_closenesses, 0, None)
        bc_closenesses = np.clip(bc_closenesses, 0, None)
        if pemirl_closenesses is not None:
            pemirl_closenesses = np.clip(pemirl_closenesses, 0, None)
        gt_baselines = np.clip(gt_baselines, 0, None)
        if sgi_closenesses is not None:
            sgi_closenesses = np.clip(sgi_closenesses, 0, None)
        if sai_closenesses is not None:
            sai_closenesses = np.clip(sai_closenesses, 0, None)

    # -------------------
    # 1) Handle the 3D square case: produce table of results
    # -------------------

    # TODO - add support fo sgi_closenesses and sai_closenesses
    if (
        len(rl_closenesses.shape) == 3
        and rl_closenesses.shape[0] == rl_closenesses.shape[1]
    ):
        with open(txt_filename, "w") as file:
            n = rl_closenesses.shape[0]
            num_trials = rl_closenesses.shape[2]

            # Build RL table
            rl_table = []
            for i in range(n):
                rl_row = []
                for j in range(n):
                    data_ij = rl_closenesses[i, j, :]
                    mean_ij = np.mean(data_ij)
                    if use_stderr:
                        se_ij = standard_error(data_ij)
                        rl_row.append(f"{mean_ij:.3f} ± {se_ij:.3f}")
                    else:
                        lower_ci, upper_ci = ci(
                            data_ij,
                            n_bootstrap=n_bootstrap,
                            ci=ci_percent,
                            studentize=studentize,
                        )
                        rl_row.append(f"{mean_ij:.3f} ({lower_ci:.3f}, {upper_ci:.3f})")
                rl_table.append(rl_row)

            # Print the RL table
            print("SRI Data Efficiency Results:")
            header = f"{y_label}    {x_label}: {x_values}"
            print(header)
            file.write(f"{header}\n")
            for i in range(n):
                row_label = f"{y_values[i]:.3f}" if y_values is not None else f"Y {i+1}"
                row_str = f"{row_label}: " + " | ".join(rl_table[i])
                print(row_str)
                file.write(f"{row_str}\n")

            # --- New: SGI table ---
            if sgi_closenesses is not None:
                # build SGI table
                sgi_table = []
                for i in range(n):
                    row = []
                    for j in range(n):
                        data_ij = sgi_closenesses[i, j, :]
                        mean_ij = np.mean(data_ij)
                        if use_stderr:
                            se_ij = standard_error(data_ij)
                            row.append(f"{mean_ij:.3f} ± {se_ij:.3f}")
                        else:
                            low, high = ci(
                                data_ij,
                                n_bootstrap=n_bootstrap,
                                ci=ci_percent,
                                studentize=studentize,
                            )
                            row.append(f"{mean_ij:.3f} ({low:.3f}, {high:.3f})")
                    sgi_table.append(row)
                print("\nSGI Data Efficiency Results:")
                file.write("\nSGI Data Efficiency Results:\n")
                for i in range(n):
                    row_label = f"{y_values[i]:.3f}" if y_values is not None else f"Y {i+1}"
                    row_str = f"{row_label}: " + " | ".join(sgi_table[i])
                    print(row_str)
                    file.write(f"{row_str}\n")

            # --- New: SAI table ---
            if sai_closenesses is not None:
                sai_table = []
                for i in range(n):
                    row = []
                    for j in range(n):
                        data_ij = sai_closenesses[i, j, :]
                        mean_ij = np.mean(data_ij)
                        if use_stderr:
                            se_ij = standard_error(data_ij)
                            row.append(f"{mean_ij:.3f} ± {se_ij:.3f}")
                        else:
                            low, high = ci(
                                data_ij,
                                n_bootstrap=n_bootstrap,
                                ci=ci_percent,
                                studentize=studentize,
                            )
                            row.append(f"{mean_ij:.3f} ({low:.3f}, {high:.3f})")
                    sai_table.append(row)
                print("\nSAI Data Efficiency Results:")
                file.write("\nSAI Data Efficiency Results:\n")
                for i in range(n):
                    row_label = f"{y_values[i]:.3f}" if y_values is not None else f"Y {i+1}"
                    row_str = f"{row_label}: " + " | ".join(sai_table[i])
                    print(row_str)
                    file.write(f"{row_str}\n")

            # Print the other methods as single rows
            print("\nAdditional Method Results:")
            file.write("\nAdditional Method Results:\n")

            # GAIL
            if not ignore_gail:
                if use_stderr:
                    gail_mean = np.mean(gail_closenesses)
                    gail_se = standard_error(gail_closenesses)
                    gail_result = f"GAIL: {gail_mean:.3f} ± {gail_se:.3f}"
                else:
                    gail_mean = np.mean(gail_closenesses)
                    gail_lower_ci, gail_upper_ci = ci(
                        gail_closenesses,
                        n_bootstrap=n_bootstrap,
                        ci=ci_percent,
                        studentize=studentize,
                    )
                    gail_result = f"GAIL: {gail_mean:.3f} ({gail_lower_ci:.3f}, {gail_upper_ci:.3f})"
                print(gail_result)
                file.write(f"{gail_result}\n")

            # AIRL
            if use_stderr:
                airl_mean = np.mean(airl_closenesses)
                airl_se = standard_error(airl_closenesses)
                airl_result = f"AIRL: {airl_mean:.3f} ± {airl_se:.3f}"
            else:
                airl_mean = np.mean(airl_closenesses)
                airl_lower_ci, airl_upper_ci = ci(
                    airl_closenesses,
                    n_bootstrap=n_bootstrap,
                    ci=ci_percent,
                    studentize=studentize,
                )
                airl_result = (
                    f"AIRL: {airl_mean:.3f} ({airl_lower_ci:.3f}, {airl_upper_ci:.3f})"
                )
            print(airl_result)
            file.write(f"{airl_result}\n")

            # BC
            if use_stderr:
                bc_mean = np.mean(bc_closenesses)
                bc_se = standard_error(bc_closenesses)
                bc_result = f"BC: {bc_mean:.3f} ± {bc_se:.3f}"
            else:
                bc_mean = np.mean(bc_closenesses)
                bc_lower_ci, bc_upper_ci = ci(
                    bc_closenesses,
                    n_bootstrap=n_bootstrap,
                    ci=ci_percent,
                    studentize=studentize,
                )
                bc_result = f"BC: {bc_mean:.3f} ({bc_lower_ci:.3f}, {bc_upper_ci:.3f})"
            print(bc_result)
            file.write(f"{bc_result}\n")

            # PEMIRL
            if pemirl_closenesses is not None:
                pemirl_summary = np.asarray(pemirl_closenesses).reshape(-1)
                if use_stderr:
                    pemirl_mean = np.mean(pemirl_summary)
                    pemirl_se = standard_error(pemirl_summary)
                    pemirl_result = f"PEMIRL: {pemirl_mean:.3f} ± {pemirl_se:.3f}"
                else:
                    pemirl_mean = np.mean(pemirl_summary)
                    pemirl_lower_ci, pemirl_upper_ci = ci(
                        pemirl_summary,
                        n_bootstrap=n_bootstrap,
                        ci=ci_percent,
                        studentize=studentize,
                    )
                    pemirl_result = (
                        f"PEMIRL: {pemirl_mean:.3f} ({pemirl_lower_ci:.3f}, {pemirl_upper_ci:.3f})"
                    )
                print(pemirl_result)
                file.write(f"{pemirl_result}\n")

            # Ground Truth Baseline
            gt_mean = np.mean(gt_baselines)
            if use_stderr:
                gt_se = standard_error(gt_baselines)
                gt_result = f"Ground Truth Baseline: {gt_mean:.3f} ± {gt_se:.3f}"
            else:
                gt_lower_ci, gt_upper_ci = ci(
                    gt_baselines,
                    n_bootstrap=n_bootstrap,
                    ci=ci_percent,
                    studentize=studentize,
                )
                gt_result = f"Ground Truth Baseline: {gt_mean:.3f} ({gt_lower_ci:.3f}, {gt_upper_ci:.3f})"
            print(gt_result)
            file.write(f"{gt_result}\n")

        return  # End 3D-square logic

    # -------------------
    # 2) Handle 1D case: print single lines
    # -------------------
    # TODO - add support for sgi_closenesses and sai_closenesses
    elif len(rl_closenesses.shape) == 1:
        with open(txt_filename, "w") as file:
            # RL
            rl_mean = np.mean(rl_closenesses)
            if use_stderr:
                rl_se = standard_error(rl_closenesses)
                rl_result = f"SRI: {rl_mean:.3f} ± {rl_se:.3f}"
            else:
                rl_low, rl_high = ci(...)
                rl_result = f"SRI: {rl_mean:.3f} ({rl_low:.3f}, {rl_high:.3f})"
            print(rl_result)
            file.write(f"{rl_result}\n")

            # --- New: SGI ---
            if sgi_closenesses is not None:
                sgi_mean = np.mean(sgi_closenesses)
                if use_stderr:
                    sgi_se = standard_error(sgi_closenesses)
                    sgi_result = f"SGI: {sgi_mean:.3f} ± {sgi_se:.3f}"
                else:
                    low, high = ci(
                        sgi_closenesses,
                        n_bootstrap=n_bootstrap,
                        ci=ci_percent,
                        studentize=studentize,
                    )
                    sgi_result = f"SGI: {sgi_mean:.3f} ({low:.3f}, {high:.3f})"
                print(sgi_result)
                file.write(f"{sgi_result}\n")

            # --- New: SAI ---
            if sai_closenesses is not None:
                sai_mean = np.mean(sai_closenesses)
                if use_stderr:
                    sai_se = standard_error(sai_closenesses)
                    sai_result = f"SAI: {sai_mean:.3f} ± {sai_se:.3f}"
                else:
                    low, high = ci(
                        sai_closenesses,
                        n_bootstrap=n_bootstrap,
                        ci=ci_percent,
                        studentize=studentize,
                    )
                    sai_result = f"SAI: {sai_mean:.3f} ({low:.3f}, {high:.3f})"
                print(sai_result)
                file.write(f"{sai_result}\n")

            # GAIL
            if not ignore_gail:
                gail_mean = np.mean(gail_closenesses)
                if use_stderr:
                    gail_se = standard_error(gail_closenesses)
                    gail_result = f"GAIL: {gail_mean:.3f} ± {gail_se:.3f}"
                else:
                    gail_lower_ci, gail_upper_ci = ci(
                        gail_closenesses,
                        n_bootstrap=n_bootstrap,
                        ci=ci_percent,
                        studentize=studentize,
                    )
                    gail_result = f"GAIL: {gail_mean:.3f} ({gail_lower_ci:.3f}, {gail_upper_ci:.3f})"
                print(gail_result)
                file.write(f"{gail_result}\n")

            # AIRL
            airl_mean = np.mean(airl_closenesses)
            if use_stderr:
                airl_se = standard_error(airl_closenesses)
                airl_result = f"AIRL: {airl_mean:.3f} ± {airl_se:.3f}"
            else:
                airl_lower_ci, airl_upper_ci = ci(
                    airl_closenesses,
                    n_bootstrap=n_bootstrap,
                    ci=ci_percent,
                    studentize=studentize,
                )
                airl_result = (
                    f"AIRL: {airl_mean:.3f} ({airl_lower_ci:.3f}, {airl_upper_ci:.3f})"
                )
            print(airl_result)
            file.write(f"{airl_result}\n")

            # BC
            bc_mean = np.mean(bc_closenesses)
            if use_stderr:
                bc_se = standard_error(bc_closenesses)
                bc_result = f"BC: {bc_mean:.3f} ± {bc_se:.3f}"
            else:
                bc_lower_ci, bc_upper_ci = ci(
                    bc_closenesses,
                    n_bootstrap=n_bootstrap,
                    ci=ci_percent,
                    studentize=studentize,
                )
                bc_result = f"BC: {bc_mean:.3f} ({bc_lower_ci:.3f}, {bc_upper_ci:.3f})"
            print(bc_result)
            file.write(f"{bc_result}\n")

            # PEMIRL
            if pemirl_closenesses is not None:
                pemirl_summary = np.asarray(pemirl_closenesses).reshape(-1)
                pemirl_mean = np.mean(pemirl_summary)
                if use_stderr:
                    pemirl_se = standard_error(pemirl_summary)
                    pemirl_result = f"PEMIRL: {pemirl_mean:.3f} ± {pemirl_se:.3f}"
                else:
                    pemirl_lower_ci, pemirl_upper_ci = ci(
                        pemirl_summary,
                        n_bootstrap=n_bootstrap,
                        ci=ci_percent,
                        studentize=studentize,
                    )
                    pemirl_result = (
                        f"PEMIRL: {pemirl_mean:.3f} ({pemirl_lower_ci:.3f}, {pemirl_upper_ci:.3f})"
                    )
                print(pemirl_result)
                file.write(f"{pemirl_result}\n")

            # Ground Truth
            gt_mean = np.mean(gt_baselines)
            if use_stderr:
                gt_se = standard_error(gt_baselines)
                gt_result = f"Ground Truth Baseline: {gt_mean:.3f} ± {gt_se:.3f}"
            else:
                gt_lower_ci, gt_upper_ci = ci(
                    gt_baselines,
                    n_bootstrap=n_bootstrap,
                    ci=ci_percent,
                    studentize=studentize,
                )
                gt_result = f"Ground Truth Baseline: {gt_mean:.3f} ({gt_lower_ci:.3f}, {gt_upper_ci:.3f})"
            print(gt_result)
            file.write(f"{gt_result}\n")
        return

    # -------------------
    # 3) For 2D case: plot line + error bars for each method
    # -------------------
    sns.set_theme(style="whitegrid")
    palette = sns.color_palette("muted")

    # Compute means
    rl_avg = np.mean(rl_closenesses, axis=1)
    if not ignore_gail:
        gail_avg = np.mean(gail_closenesses, axis=1)
    airl_avg = np.mean(airl_closenesses, axis=1)
    bc_avg = np.mean(bc_closenesses, axis=1)
    if pemirl_closenesses is not None:
        pemirl_avg = np.mean(pemirl_closenesses, axis=1)
    gt_avg = np.mean(gt_baselines)
    if sgi_closenesses is not None:
        sgi_avg = np.mean(sgi_closenesses, axis=1)
    if sai_closenesses is not None:
        sai_avg = np.mean(sai_closenesses, axis=1)

    if clip_avg_to_zero:
        rl_avg = np.clip(rl_avg, 0, None)
        if not ignore_gail:
            gail_avg = np.clip(gail_avg, 0, None)
        airl_avg = np.clip(airl_avg, 0, None)
        bc_avg = np.clip(bc_avg, 0, None)
        if pemirl_closenesses is not None:
            pemirl_avg = np.clip(pemirl_avg, 0, None)
        gt_avg = np.clip(gt_avg, 0, None)
        if sgi_closenesses is not None:
            sgi_avg = np.clip(sgi_avg, 0, None)
        if sai_closenesses is not None:
            sai_avg = np.clip(sai_avg, 0, None)

    # Prepare arrays for error bars
    if use_stderr:
        # Just compute SE for each x-value
        rl_se = np.std(rl_closenesses, axis=1, ddof=1) / np.sqrt(
            rl_closenesses.shape[1]
        )
        if not ignore_gail:
            gail_se = np.std(gail_closenesses, axis=1, ddof=1) / np.sqrt(
                gail_closenesses.shape[1]
            )
        airl_se = np.std(airl_closenesses, axis=1, ddof=1) / np.sqrt(
            airl_closenesses.shape[1]
        )
        bc_se = np.std(bc_closenesses, axis=1, ddof=1) / np.sqrt(
            bc_closenesses.shape[1]
        )
        if pemirl_closenesses is not None:
            pemirl_se = np.std(pemirl_closenesses, axis=1, ddof=1) / np.sqrt(
                pemirl_closenesses.shape[1]
            )
        if sgi_closenesses is not None:
            sgi_se = np.std(sgi_closenesses, axis=1, ddof=1) / np.sqrt(
                sgi_closenesses.shape[1]
            )
        if sai_closenesses is not None:
            sai_se = np.std(sai_closenesses, axis=1, ddof=1) / np.sqrt(
                sai_closenesses.shape[1]
            )
        # For GT (horizontal line), we won't do an error band, but you could if desired
        gt_se_val = standard_error(gt_baselines)
    else:
        # Use bootstrap CI for each x-value
        rl_ci_low, rl_ci_high = [], []
        if not ignore_gail:
            gail_ci_low, gail_ci_high = [], []
        airl_ci_low, airl_ci_high = [], []
        bc_ci_low, bc_ci_high = [], []
        if pemirl_closenesses is not None:
            pemirl_ci_low, pemirl_ci_high = [], []
        if sgi_closenesses is not None:
            sgi_ci_low, sgi_ci_high = [], []
        if sai_closenesses is not None:
            sai_ci_low, sai_ci_high = [], []

        for i in range(len(x_values)):
            # RL
            lower, upper = ci(
                rl_closenesses[i, :],
                n_bootstrap=n_bootstrap,
                ci=ci_percent,
                studentize=studentize,
            )
            rl_ci_low.append(lower)
            rl_ci_high.append(upper)

            # GAIL
            if not ignore_gail:
                g_lower, g_upper = ci(
                    gail_closenesses[i, :],
                    n_bootstrap=n_bootstrap,
                    ci=ci_percent,
                    studentize=studentize,
                )
                gail_ci_low.append(g_lower)
                gail_ci_high.append(g_upper)

            # AIRL
            a_lower, a_upper = ci(
                airl_closenesses[i, :],
                n_bootstrap=n_bootstrap,
                ci=ci_percent,
                studentize=studentize,
            )
            airl_ci_low.append(a_lower)
            airl_ci_high.append(a_upper)

            # BC
            b_lower, b_upper = ci(
                bc_closenesses[i, :],
                n_bootstrap=n_bootstrap,
                ci=ci_percent,
                studentize=studentize,
            )
            bc_ci_low.append(b_lower)
            bc_ci_high.append(b_upper)

            # PEMIRL
            if pemirl_closenesses is not None:
                p_lower, p_upper = ci(
                    pemirl_closenesses[i, :],
                    n_bootstrap=n_bootstrap,
                    ci=ci_percent,
                    studentize=studentize,
                )
                pemirl_ci_low.append(p_lower)
                pemirl_ci_high.append(p_upper)

            # SGI
            if sgi_closenesses is not None:
                s_lower, s_upper = ci(
                    sgi_closenesses[i, :],
                    n_bootstrap=n_bootstrap,
                    ci=ci_percent,
                    studentize=studentize,
                )
                sgi_ci_low.append(s_lower)
                sgi_ci_high.append(s_upper)
            # SAI
            if sai_closenesses is not None:
                sa_lower, sa_upper = ci(
                    sai_closenesses[i, :],
                    n_bootstrap=n_bootstrap,
                    ci=ci_percent,
                    studentize=studentize,
                )
                sai_ci_low.append(sa_lower)
                sai_ci_high.append(sa_upper)

        rl_ci_low, rl_ci_high = np.array(rl_ci_low), np.array(rl_ci_high)
        if not ignore_gail:
            gail_ci_low, gail_ci_high = np.array(gail_ci_low), np.array(gail_ci_high)
        airl_ci_low, airl_ci_high = np.array(airl_ci_low), np.array(airl_ci_high)
        bc_ci_low, bc_ci_high = np.array(bc_ci_low), np.array(bc_ci_high)
        if pemirl_closenesses is not None:
            pemirl_ci_low, pemirl_ci_high = (
                np.array(pemirl_ci_low),
                np.array(pemirl_ci_high),
            )
        if sgi_closenesses is not None:
            sgi_ci_low, sgi_ci_high = np.array(sgi_ci_low), np.array(sgi_ci_high)
        if sai_closenesses is not None:
            sai_ci_low, sai_ci_high = np.array(sai_ci_low), np.array(sai_ci_high)

        # For GT
        gt_ci_low, gt_ci_high = ci(
            gt_baselines,
            n_bootstrap=n_bootstrap,
            ci=ci_percent,
            studentize=studentize,
        )

    # plt.figure(figsize=(9, 7), layout="constrained")
    plt.figure(figsize=(7, 6), layout="constrained")

    if equal_spacing:
        x_values_plot = np.linspace(0, 1, len(x_values))
    else:
        x_values_plot = x_values

    # --- Plot each method with either symmetrical SE or (lower, upper) from bootstrap
    if use_stderr:
        # error_kw_1 = dict(ecolor="C0", linewidth=2.0, capsize=4)
        # error_kw_1 = dict(ecolor=palette[0], linewidth=2.0, capsize=4)
        plt.errorbar(
            x_values_plot,
            rl_avg,
            yerr=rl_se,
            label="SRI",
            marker="*",
            capsize=5,
            color=palette[0],
            linestyle="-",
            alpha=0.8,
            linewidth=3,
            # error_kw=error_kw_1,
            elinewidth=3,
        )
        if sgi_closenesses is not None:
            # error_kw_2 = dict(ecolor=palette[1], linewidth=2.0, capsize=4)
            plt.errorbar(
                x_values_plot,
                sgi_avg,
                yerr=sgi_se,
                label="SGI",
                marker="o",
                capsize=5,
                # color=palette[0],
                color=palette[5],
                linestyle="-",
                alpha=0.8,
                linewidth=3,
                # error_kw=error_kw_2,
                elinewidth=3,
            )
        if sai_closenesses is not None:
            # error_kw_3 = dict(ecolor=palette[2], linewidth=2.0, capsize=4)
            plt.errorbar(
                x_values_plot,
                sai_avg,
                yerr=sai_se,
                label="SAI",
                marker="v",
                capsize=5,
                # color=palette[0],
                color=palette[6],
                linestyle="-",
                alpha=0.8,
                linewidth=3,
                # error_kw=error_kw_3,
                elinewidth=3,
            )
        if not ignore_gail:
            # error_kw_2 = dict(ecolor=palette[1], linewidth=2.0, capsize=4)
            plt.errorbar(
                x_values_plot,
                gail_avg,
                yerr=gail_se,
                label="GAIL",
                marker="s",
                capsize=5,
                color=palette[1],
                linestyle="--",
                alpha=0.8,
                linewidth=3,
                # error_kw=error_kw_2,
                elinewidth=3,
            )
        # error_kw_3 = dict(ecolor=palette[2], linewidth=2.0, capsize=4)
        plt.errorbar(
            x_values_plot,
            airl_avg,
            yerr=airl_se,
            label="AIRL",
            marker="^",
            capsize=5,
            color=palette[2],
            linestyle="-.",
            alpha=0.8,
            linewidth=3,
            # error_kw=error_kw_3,
            elinewidth=3,
        )
        # error_kw_4 = dict(ecolor=palette[3], linewidth=2.0, capsize=4)
        plt.errorbar(
            x_values_plot,
            bc_avg,
            yerr=bc_se,
            label="BC",
            marker="d",
            capsize=5,
            color=palette[3],
            linestyle=":",
            alpha=0.8,
            linewidth=3,
            # error_kw=error_kw_4,
            elinewidth=3,
        )
        if pemirl_closenesses is not None:
            plt.errorbar(
                x_values_plot,
                pemirl_avg,
                yerr=pemirl_se,
                label="PEMIRL",
                marker="P",
                capsize=5,
                color=palette[7],
                linestyle="-",
                alpha=0.8,
                linewidth=3,
                elinewidth=3,
            )
        # For GT we just plot a line at the mean
        plt.axhline(
            y=gt_avg,
            color="black",
            linestyle="--",
            label="GT RL",
            linewidth=2,
        )
        plt.axhline(
            y=0.0,
            color=palette[4],
            linestyle=":",
            label="No op",
            linewidth=3,
        )
    else:
        # Plot with bootstrap intervals
        plt.errorbar(
            x_values_plot,
            rl_avg,
            yerr=[rl_avg - rl_ci_low, rl_ci_high - rl_avg],
            label="SRI",
            marker="*",
            capsize=5,
            color=palette[0],
            linestyle="-",
            alpha=0.8,
        )
        if sgi_closenesses is not None:
            plt.errorbar(
                x_values_plot,
                sgi_avg,
                yerr=[sgi_avg - sgi_ci_low, sgi_ci_high - sgi_avg],
                label="SGI",
                marker="o",
                capsize=5,
                color=palette[5],
                linestyle="-",
                alpha=0.8,
            )
        if sai_closenesses is not None:
            plt.errorbar(
                x_values_plot,
                sai_avg,
                yerr=[sai_avg - sai_ci_low, sai_ci_high - sai_avg],
                label="SAI",
                marker="v",
                capsize=5,
                color=palette[6],
                linestyle="-",
                alpha=0.8,
            )
        if not ignore_gail:
            plt.errorbar(
                x_values_plot,
                gail_avg,
                yerr=[gail_avg - gail_ci_low, gail_ci_high - gail_avg],
                label="GAIL",
                marker="s",
                capsize=5,
                color=palette[1],
                linestyle="--",
                alpha=0.8,
            )
        plt.errorbar(
            x_values_plot,
            airl_avg,
            yerr=[airl_avg - airl_ci_low, airl_ci_high - airl_avg],
            label="AIRL",
            marker="^",
            capsize=5,
            color=palette[2],
            linestyle="-.",
            alpha=0.8,
        )
        plt.errorbar(
            x_values_plot,
            bc_avg,
            yerr=[bc_avg - bc_ci_low, bc_ci_high - bc_avg],
            label="BC",
            marker="d",
            capsize=5,
            color=palette[3],
            linestyle=":",
            alpha=0.8,
        )
        if pemirl_closenesses is not None:
            plt.errorbar(
                x_values_plot,
                pemirl_avg,
                yerr=[pemirl_avg - pemirl_ci_low, pemirl_ci_high - pemirl_avg],
                label="PEMIRL",
                marker="P",
                capsize=5,
                color=palette[7],
                linestyle="-",
                alpha=0.8,
            )
        plt.axhline(
            y=gt_avg,
            color="black",
            linestyle="--",
            label="GT RL",
            linewidth=1.5,
        )

    if log_x:
        plt.xscale("log")
    if reverse_x:
        plt.gca().invert_xaxis()

    plt.xticks(ticks=x_values_plot, labels=x_values, fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel(x_label, fontsize=22)
    plt.ylabel(y_label, fontsize=22)
    # if y_axis_start_at_zero:
    if y_axis_min is not None:
        # plt.ylim(bottom=0)
        plt.ylim(bottom=y_axis_min)

    plt.legend(loc="best", frameon=True, fancybox=True, shadow=True, fontsize=16)

    plt.savefig(filename, format="pdf", bbox_inches="tight")

    # Write text file (just GT info here for example, you can add more if desired)
    auto_txt_filename = TABLES_DIR / (Path(filename).stem + ".txt")
    with open(auto_txt_filename, "w") as file:
        file.write(f"Ground Truth Baseline Mean: {gt_avg}\n")
        if use_stderr:
            file.write(f"Ground Truth Baseline ± SE: ±{gt_se_val}\n")
        else:
            file.write(f"Ground Truth Baseline CI: ({gt_ci_low}, {gt_ci_high})\n")


def ci(data, n_bootstrap=1000, ci=95, studentize=False, method="bootstrap"):
    original_mean = np.mean(data)
    n_data = len(data)

    if method == "bootstrap":
        bootstrap_means = np.zeros(n_bootstrap)
        if studentize:
            bootstrap_se = np.zeros(n_bootstrap)

        for i in range(n_bootstrap):
            bootstrap_sample = np.random.choice(data, size=n_data, replace=True)
            bootstrap_means[i] = np.mean(bootstrap_sample)

            if studentize:
                bootstrap_se[i] = np.std(bootstrap_sample, ddof=1) / np.sqrt(
                    len(bootstrap_sample)
                )

        if studentize:
            t_stats = (bootstrap_means - original_mean) / bootstrap_se
            lower_percentile = np.percentile(t_stats, (100 - ci) / 2)
            upper_percentile = np.percentile(t_stats, 100 - (100 - ci) / 2)
            se_original = np.std(data, ddof=1) / np.sqrt(len(data))
            lower = original_mean + lower_percentile * se_original
            upper = original_mean + upper_percentile * se_original
        else:
            lower = np.percentile(bootstrap_means, (100 - ci) / 2)
            upper = np.percentile(bootstrap_means, 100 - (100 - ci) / 2)

    elif method == "hoeffding":
        delta = np.sqrt(np.log(2 / ((100 - ci) / 100)) / (2 * n_data))
        lower = original_mean - delta
        upper = original_mean + delta

    else:
        raise ValueError(
            "Unsupported method specified. Choose 'bootstrap' or 'hoeffding'."
        )

    return lower, upper



def get_tst_ablation_results():
    """
    Processes experiments for SRI, SGI, SAI with TST (default) and Transformer encoders.
    """
    model_idxs = list(range(30)) # Assuming 30 trials as in get_n_results
    noise_coeff = 0.87 # As in get_n_results
    train_configs_paths = ["train/n1_skip.yml", "train/n10_skip.yml", "train/skip.yml"]
    print(f"Train configurations: {train_configs_paths}")

    # Common configurations
    sri_general_config_path = "general/default_experiment.yml"
    sri_dataset_config_path = "datasets/reach_noise_fixed.yml"
    sri_obs_dataset_config_path = "datasets/reach_for_pickplace.yml"
    sri_model_config_default_path = "model/load_default.yml" # Assumed to be TST by default
    sri_rl_config_path = "rl/tqc_quarter_legacy_paper_esr_5m.yml"
    sri_inference_config_path = "inference/eval_10_envs.yml" # For SRI, SGI, SAI
    # Baseline config is needed by process_configs, even if not used for evaluation type
    default_baseline_config_path = "baselines/default.yml"

    sgi_general_config_path = "general/default_eval.yml"
    sgi_dataset_config_path = "datasets/reach_noise_fixed.yml"
    sgi_obs_dataset_config_path = "datasets/reach_for_pickplace.yml"
    sgi_model_config_default_path = "model/goals.yml" # Assumed to be TST by default
    sgi_rl_config_path = "rl/tqc_quarter_legacy_paper_esr_5m.yml"
    # sgi_inference_config_path = "inference/eval_20_envs.yml" # From get_n_results, using sri_inference_config_path for consistency or choose one
    
    sai_general_config_path = "general/default_eval.yml"
    sai_dataset_config_path = "datasets/reach_noise_fixed.yml"
    sai_obs_dataset_config_path = "datasets/reach_for_pickplace.yml"
    sai_model_config_default_path = "model/actions.yml" # Assumed to be TST by default
    sai_rl_config_path = "rl/tqc_quarter_legacy_paper_esr_5m.yml"
    # sai_inference_config_path = "inference/eval_20_envs.yml" # From get_n_results

    # Load general configs once
    sri_general_config = load_config_with_defaults(sri_general_config_path)
    sri_dataset_config_base = load_config_with_defaults(sri_dataset_config_path)
    sri_obs_dataset_config = load_config_with_defaults(sri_obs_dataset_config_path)
    sri_rl_config = load_config_with_defaults(sri_rl_config_path)
    sri_inference_config = load_config_with_defaults(sri_inference_config_path) # Use this for all
    default_baseline_config = load_config_with_defaults(default_baseline_config_path)

    sri_dataset_config_base.noise_coeff = noise_coeff
    
    # SGI specific configs (general, rl, inference might be shared or specific)
    sgi_general_config = load_config_with_defaults(sgi_general_config_path)
    sgi_dataset_config_base = load_config_with_defaults(sgi_dataset_config_path)
    sgi_obs_dataset_config = load_config_with_defaults(sgi_obs_dataset_config_path)
    sgi_rl_config = load_config_with_defaults(sgi_rl_config_path)
    # Adjust SGI specific parameters as in get_n_results
    sgi_dataset_config_base.noise_coeff = noise_coeff
    # sgi_inference_config = load_config_with_defaults("inference/eval_20_envs.yml") # if different
    # sgi_inference_config.batch_size = 2
    # sgi_inference_config.episodes = 50
    sgi_rl_config.no_task_rep = False


    # SAI specific configs
    sai_general_config = load_config_with_defaults(sai_general_config_path)
    sai_dataset_config_base = load_config_with_defaults(sai_dataset_config_path)
    sai_obs_dataset_config = load_config_with_defaults(sai_obs_dataset_config_path)
    sai_rl_config = load_config_with_defaults(sai_rl_config_path)
    # Adjust SAI specific parameters as in get_n_results
    sai_dataset_config_base.noise_coeff = noise_coeff
    # sai_inference_config = load_config_with_defaults("inference/eval_20_envs.yml") # if different
    # sai_inference_config.batch_size = 2
    # sai_inference_config.episodes = 50
    # sai_inference_config.include_extra_reward_info = True # If needed for SAI
    sai_rl_config.no_task_rep = False


    results_sri_tst = []
    results_sri_transformer = []
    results_sgi_tst = []
    results_sgi_transformer = []
    results_sai_tst = []
    results_sai_transformer = []

    for train_config_path in tqdm(train_configs_paths, desc="Processing train configurations for TST ablation"):
        train_config = load_config_with_defaults(train_config_path)
        # Potentially set train_config.num_epochs = 500 if needed for SGI/SAI,
        # though get_n_results sets it after process_configs for SGI/SAI.
        # If process_configs runs the experiment, it should be set before.
        # For now, assuming it's handled or not strictly needed for this ablation.

        # --- SRI ---
        # SRI TST (Default)
        print(f"Processing SRI-TST with {train_config_path}")
        sri_model_config_tst = load_config_with_defaults(sri_model_config_default_path)
        process_configs(
            sri_general_config, sri_dataset_config_base, sri_obs_dataset_config,
            sri_model_config_tst, train_config, sri_rl_config,
            sri_inference_config, default_baseline_config
        )
        results_sri_tst.append(get_eval_results(
            sri_general_config, sri_model_config_tst, train_config,
            sri_dataset_config_base, sri_obs_dataset_config, most_recent_first=True,
            model_idxs=model_idxs, rl=True, rl_config=sri_rl_config, baseline=False,
            inference_config=sri_inference_config
        ))

        # SRI Transformer
        print(f"Processing SRI-Transformer with {train_config_path}")
        sri_model_config_transformer = load_config_with_defaults(sri_model_config_default_path)
        sri_model_config_transformer.dem_encoder_type = "transformer"
        process_configs(
            sri_general_config, sri_dataset_config_base, sri_obs_dataset_config,
            sri_model_config_transformer, train_config, sri_rl_config,
            sri_inference_config, default_baseline_config
        )
        results_sri_transformer.append(get_eval_results(
            sri_general_config, sri_model_config_transformer, train_config,
            sri_dataset_config_base, sri_obs_dataset_config, most_recent_first=True,
            model_idxs=model_idxs, rl=True, rl_config=sri_rl_config, baseline=False,
            inference_config=sri_inference_config
        ))

        # --- SGI ---
        current_train_config_sgi_sai = load_config_with_defaults(train_config_path) # Fresh load for SGI/SAI
        if hasattr(current_train_config_sgi_sai, 'num_epochs'): # As per get_n_results structure for SGI/SAI
             current_train_config_sgi_sai.num_epochs = 500

        # SGI TST (Default)
        print(f"Processing SGI-TST with {train_config_path}")
        sgi_model_config_tst = load_config_with_defaults(sgi_model_config_default_path)
        if hasattr(sgi_model_config_tst, 'load_model'): sgi_model_config_tst.load_model = True # As per get_n_results SGI/SAI
        process_configs(
            sgi_general_config, sgi_dataset_config_base, sgi_obs_dataset_config,
            sgi_model_config_tst, current_train_config_sgi_sai, sgi_rl_config,
            sri_inference_config, default_baseline_config # Using common sri_inference_config
        )
        results_sgi_tst.append(get_eval_results(
            sgi_general_config, sgi_model_config_tst, current_train_config_sgi_sai,
            sgi_dataset_config_base, sgi_obs_dataset_config, most_recent_first=True,
            model_idxs=model_idxs, rl=False, rl_config=sgi_rl_config, baseline=False, # rl=False for SGI/SAI from get_n_results
            inference_config=sri_inference_config
        ))

        # SGI Transformer
        print(f"Processing SGI-Transformer with {train_config_path}")
        sgi_model_config_transformer = load_config_with_defaults(sgi_model_config_default_path)
        if hasattr(sgi_model_config_transformer, 'load_model'): sgi_model_config_transformer.load_model = True
        sgi_model_config_transformer.dem_encoder_type = "transformer"
        process_configs(
            sgi_general_config, sgi_dataset_config_base, sgi_obs_dataset_config,
            sgi_model_config_transformer, current_train_config_sgi_sai, sgi_rl_config,
            sri_inference_config, default_baseline_config
        )
        results_sgi_transformer.append(get_eval_results(
            sgi_general_config, sgi_model_config_transformer, current_train_config_sgi_sai,
            sgi_dataset_config_base, sgi_obs_dataset_config, most_recent_first=True,
            model_idxs=model_idxs, rl=False, rl_config=sgi_rl_config, baseline=False,
            inference_config=sri_inference_config
        ))

        # --- SAI ---
        # SAI TST (Default)
        print(f"Processing SAI-TST with {train_config_path}")
        sai_model_config_tst = load_config_with_defaults(sai_model_config_default_path)
        if hasattr(sai_model_config_tst, 'load_model'): sai_model_config_tst.load_model = True
        process_configs(
            sai_general_config, sai_dataset_config_base, sai_obs_dataset_config,
            sai_model_config_tst, current_train_config_sgi_sai, sai_rl_config, # Re-using current_train_config_sgi_sai
            sri_inference_config, default_baseline_config
        )
        results_sai_tst.append(get_eval_results(
            sai_general_config, sai_model_config_tst, current_train_config_sgi_sai,
            sai_dataset_config_base, sai_obs_dataset_config, most_recent_first=True,
            model_idxs=model_idxs, rl=False, rl_config=sai_rl_config, baseline=False,
            inference_config=sri_inference_config
        ))

        # SAI Transformer
        print(f"Processing SAI-Transformer with {train_config_path}")
        sai_model_config_transformer = load_config_with_defaults(sai_model_config_default_path)
        if hasattr(sai_model_config_transformer, 'load_model'): sai_model_config_transformer.load_model = True
        sai_model_config_transformer.dem_encoder_type = "transformer"
        process_configs(
            sai_general_config, sai_dataset_config_base, sai_obs_dataset_config,
            sai_model_config_transformer, current_train_config_sgi_sai, sai_rl_config,
            sri_inference_config, default_baseline_config
        )
        results_sai_transformer.append(get_eval_results(
            sai_general_config, sai_model_config_transformer, current_train_config_sgi_sai,
            sai_dataset_config_base, sai_obs_dataset_config, most_recent_first=True,
            model_idxs=model_idxs, rl=False, rl_config=sai_rl_config, baseline=False,
            inference_config=sri_inference_config
        ))

    print("TST Ablation evaluation complete.")
    return (
        np.array(results_sri_tst), np.array(results_sri_transformer),
        np.array(results_sgi_tst), np.array(results_sgi_transformer),
        np.array(results_sai_tst), np.array(results_sai_transformer)
    )


def plot_tst_ablation_lines(
    x_values,
    sri_tst_data, sgi_tst_data, sai_tst_data,
    sri_transformer_data, sgi_transformer_data, sai_transformer_data,
    filename=str(PLOTS_DIR / "tst_ablation_plot.pdf"),
    log_x=True,
    x_label="Number of Inference Demonstrations",
    y_label="Average Scaled Goal Proximity",
    use_stderr=True,
    n_bootstrap=1000, ci_percent=95, studentize=False, # For CI if use_stderr is False
    clip_to_zero=True, # clip individual trials before mean/se/ci
    clip_avg_to_zero=False, # clip mean values before plotting
    y_axis_min=-0.05,
    palette=None
):
    """
    Plots the TST vs Transformer ablation results.
    Each data array is expected to be of shape (num_x_values, num_trials).
    """
    ensure_output_dirs()
    filename = str(filename)

    if palette is None:
        palette = sns.color_palette("muted")

    all_data = {
        "SRI": sri_tst_data, "SGI": sgi_tst_data, "SAI": sai_tst_data,
        "SRI-transformer-sum": sri_transformer_data,
        "SGI-transformer-sum": sgi_transformer_data,
        "SAI-transformer-sum": sai_transformer_data,
    }

    if clip_to_zero:
        for key in all_data:
            all_data[key] = np.clip(all_data[key], 0, None)

    means = {}
    errors_low = {} # Stores lower part of error bar (e.g., mean - SE or mean - CI_low)
    errors_high = {} # Stores upper part of error bar (e.g., SE or CI_high - mean)

    for i, (label, data) in enumerate(all_data.items()):
        means[label] = np.mean(data, axis=1)
        if clip_avg_to_zero:
            means[label] = np.clip(means[label], 0, None)

        if use_stderr:
            se = standard_error(data, axis=-1)
            errors_low[label] = se # For yerr, SE is symmetrical
            errors_high[label] = se
        else:
            ci_low_vals, ci_high_vals = [], []
            for x_idx in range(data.shape[0]):
                low, high = ci(data[x_idx, :], n_bootstrap=n_bootstrap, ci=ci_percent, studentize=studentize)
                ci_low_vals.append(low)
                ci_high_vals.append(high)
            errors_low[label] = means[label] - np.array(ci_low_vals)
            errors_high[label] = np.array(ci_high_vals) - means[label]


    plt.figure(figsize=(8, 6), layout="constrained") # Adjusted size slightly
    sns.set_theme(style="whitegrid")

    # Define colors and styles
    # SRI: palette[0], SGI: palette[1], SAI: palette[2] (example)
    # You might want to align these with colors used in plot_and_save_results_with_ci_stylish
    color_map = {
        "SRI": palette[0], "SGI": palette[1], "SAI": palette[2],
        "SRI-transformer-sum": palette[0],
        "SGI-transformer-sum": palette[1],
        "SAI-transformer-sum": palette[2],
    }
    linestyle_map = {
        "SRI": "-", "SGI": "-", "SAI": "-",
        "SRI-transformer-sum": ":", # Dotted
        "SGI-transformer-sum": ":",
        "SAI-transformer-sum": ":",
    }
    marker_map = { # Consistent markers for SRI, SGI, SAI
        "SRI": "*", "SGI": "o", "SAI": "s",
        "SRI-transformer-sum": "*",
        "SGI-transformer-sum": "o",
        "SAI-transformer-sum": "s",
    }


    x_values_plot = np.array(x_values)

    for label in means:
        if use_stderr:
            # yerr expects a single value for symmetric errors, or (below, above) for asymmetric
            yerr_values = errors_low[label] # since errors_low[label] == errors_high[label] == SE
        else:
            yerr_values = [errors_low[label], errors_high[label]]

        plt.errorbar(
            x_values_plot, means[label], yerr=yerr_values, label=label,
            marker=marker_map.get(label, 'x'), capsize=5, color=color_map.get(label, palette[3]),
            linestyle=linestyle_map.get(label, '-'), alpha=0.9, linewidth=2.5, elinewidth=2.5
        )

    if log_x:
        plt.xscale("log")

    plt.xticks(ticks=x_values_plot, labels=[str(x) for x in x_values], fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel(x_label, fontsize=20) # Adjusted font size
    plt.ylabel(y_label, fontsize=20) # Adjusted font size

    if y_axis_min is not None:
        plt.ylim(bottom=y_axis_min)

    plt.legend(loc="best", frameon=True, fancybox=True, shadow=False, fontsize=14) # Adjusted font size
    plt.savefig(filename, format="pdf", bbox_inches="tight")
    print(f"Saved TST ablation plot to {filename}")
    plt.close()


def process_tst_ablation(use_cache=False):
    print("Starting TST ablation processing...")
    ensure_output_dirs()
    cache_path = CACHE_DIR / "tst_ablation_results.npz"

    if use_cache and os.path.exists(cache_path):
        print("Loading cached TST ablation results...")
        results_data = np.load(cache_path)
        sri_tst_closenesses = results_data["sri_tst_closenesses"]
        sri_transformer_closenesses = results_data["sri_transformer_closenesses"]
        sgi_tst_closenesses = results_data["sgi_tst_closenesses"]
        sgi_transformer_closenesses = results_data["sgi_transformer_closenesses"]
        sai_tst_closenesses = results_data["sai_tst_closenesses"]
        sai_transformer_closenesses = results_data["sai_transformer_closenesses"]
        results_data.close()
    else:
        print("Running experiments to get TST ablation results...")
        (sri_tst_closenesses, sri_transformer_closenesses,
         sgi_tst_closenesses, sgi_transformer_closenesses,
         sai_tst_closenesses, sai_transformer_closenesses) = get_tst_ablation_results()

        if use_cache: # Save even if use_cache was initially False but we ran exps
             print("Saving TST ablation results to cache...")
             np.savez(
                cache_path,
                sri_tst_closenesses=sri_tst_closenesses,
                sri_transformer_closenesses=sri_transformer_closenesses,
                sgi_tst_closenesses=sgi_tst_closenesses,
                sgi_transformer_closenesses=sgi_transformer_closenesses,
                sai_tst_closenesses=sai_tst_closenesses,
                sai_transformer_closenesses=sai_transformer_closenesses,
            )

    print("Plotting TST ablation results...")
    plot_tst_ablation_lines(
        x_values=[1, 10, 100],
        sri_tst_data=sri_tst_closenesses,
        sgi_tst_data=sgi_tst_closenesses,
        sai_tst_data=sai_tst_closenesses,
        sri_transformer_data=sri_transformer_closenesses,
        sgi_transformer_data=sgi_transformer_closenesses,
        sai_transformer_data=sai_transformer_closenesses,
        filename=str(PLOTS_DIR / "tst_transformer_ablation_n_plot.pdf"), # Specific filename
        log_x=True,
        x_label="Number of Inference Demonstrations (n)",
        y_label="Average Scaled Goal Proximity",
        # Using defaults from plot_tst_ablation_lines for use_stderr, clip, y_axis_min
    )
    print("TST ablation processing complete.")




def load_reach_gt_baseline_results():
    # python -m sri.run_experiment_temp_2 \
    #     --general-config general/default_eval.yml \
    #     --train-config train/skip.yml \
    #     --dataset-config datasets/reach_noise.yml \
    #     --obs-dataset-config datasets/reach_noise.yml \
    #     --orl-dataset-config datasets/reach_noise.yml \
    #     --model-config model/load_default.yml \
    #     --rl-config rl/tqc_quarter_legacy_paper_esr_5m_gt_reward.yml \
    #     --inference-config inference/eval_10_envs.yml \
    #     --model-idxs $MODEL_IDX
    model_idxs = list(range(num_runs))
    general_config = "general/default_experiment.yml"
    train_config = "train/skip.yml"
    dataset_config = "datasets/reach_noise.yml"
    obs_dataset_config = "datasets/reach_noise.yml"
    model_config = "model/load_default.yml"
    rl_config = "rl/tqc_quarter_legacy_paper_esr_5m_gt_reward.yml"
    inference_config = "inference/eval_10_envs.yml"
    baselines_config = "baselines/default.yml"

    print("Loading configurations...")
    general_config = load_config_with_defaults(general_config)
    dataset_config = load_config_with_defaults(dataset_config)
    obs_dataset_config = load_config_with_defaults(obs_dataset_config)
    model_config = load_config_with_defaults(model_config)
    train_config = load_config_with_defaults(train_config)
    rl_config = load_config_with_defaults(rl_config)
    inference_config = load_config_with_defaults(inference_config)
    baselines_config = load_config_with_defaults(baselines_config)

    print("Processing configurations...")
    process_configs(
        general_config,
        dataset_config,
        obs_dataset_config,
        model_config,
        train_config,
        rl_config,
        inference_config,
        baselines_config,
    )

    print("Getting evaluation results...")
    result = get_eval_results(
        general_config,
        model_config,
        train_config,
        dataset_config,
        obs_dataset_config,
        most_recent_first=True,
        model_idxs=model_idxs,
        rl=True,
        rl_config=rl_config,
        baseline=False,
        inference_config=inference_config,
    )
    return np.array(result)

def process_reach_gt_baseline(num_runs=10, use_cache=False,):
    print("Starting process_reach_gt_baseline function...")
    
    ensure_output_dirs()
    cache_path = CACHE_DIR / "reach_gt_baseline_results.npz"
    if use_cache and os.path.exists(cache_path):
        print("Loading cached results...")
        results = np.load(cache_path)
        rl_closenesses = results["rl_closenesses"]
        results.close()
    else:
        rl_closenesses = load_reach_gt_baseline_results()
        # if use_cache:
        print("Saving results to cache...")
        np.savez(cache_path, rl_closenesses=rl_closenesses)

    print("Evaluation complete.")
    return rl_closenesses

def load_pickplace_gt_baseline_results():
    # python -m sri.run_experiment \
    #     --general-config general/default_experiment.yml \
    #     --train-config train/pickplace_more_obs_skip.yml \
    #     --dataset-config datasets/reach_avoid_obj.yml \
    #     --obs-dataset-config datasets/default_pickplace.yml \
    #     --obs-dataset-config-2 datasets/reach_for_pickplace.yml \
    #     --model-config model/load_default.yml \
    #     --rl-config rl/ppo_quarter_legacy_paper_esr_no_goal_gt_reward.yml \
    #     --inference-config inference/pickplace_single_only_rl_reinit_obj_pos.yml \
    #     --dataset-idx $DATASET_IDX \
    #     --obs-dataset-idx $DATASET_IDX \
    #     --obs-dataset-2-idx $DATASET_IDX \
    #     --model-idxs $MODEL_IDX \
    model_idxs = list(range(num_runs))

    general_config = "general/default_experiment.yml"
    train_config = "train/pickplace_more_obs_skip.yml"
    dataset_config = "datasets/reach_avoid_obj.yml"
    obs_dataset_config = "datasets/default_pickplace.yml"
    model_config = "model/load_default.yml"
    rl_config = "rl/ppo_quarter_legacy_paper_esr_no_goal_gt_reward.yml"
    inference_config = "inference/pickplace_single_only_rl_reinit_obj_pos.yml"
    baselines_config = "baselines/default.yml"

    print("Loading configurations...")
    general_config = load_config_with_defaults(general_config)
    dataset_config = load_config_with_defaults(dataset_config)
    obs_dataset_config = load_config_with_defaults(obs_dataset_config)
    model_config = load_config_with_defaults(model_config)
    train_config = load_config_with_defaults(train_config)
    rl_config = load_config_with_defaults(rl_config)
    inference_config = load_config_with_defaults(inference_config)
    baselines_config = load_config_with_defaults(baselines_config)

    print("Processing configurations...")
    process_configs(
        general_config,
        dataset_config,
        obs_dataset_config,
        model_config,
        train_config,
        rl_config,
        inference_config,
        baselines_config,
    )

    print("Getting evaluation results...")
    result = get_eval_results(
        general_config,
        model_config,
        train_config,
        dataset_config,
        obs_dataset_config,
        most_recent_first=True,
        model_idxs=model_idxs,
        rl=True,
        rl_config=rl_config,
        baseline=False,
        inference_config=inference_config,
    )
    return np.array(result)

def process_pickplace_gt_baseline(num_runs=10, use_cache=False,):
    print("Starting process_pickplace_gt_baseline function...")
    
    ensure_output_dirs()
    cache_path = CACHE_DIR / "pickplace_gt_baseline_results.npz"
    if use_cache and os.path.exists(cache_path):
        print("Loading cached results...")
        results = np.load(cache_path)
        rl_closenesses = results["rl_closenesses"]
        results.close()
    else:
        rl_closenesses = load_pickplace_gt_baseline_results()
        # if use_cache:
        print("Saving results to cache...")
        np.savez(cache_path, rl_closenesses=rl_closenesses)

    print("Evaluation complete.")
    return rl_closenesses


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("exp_to_process", type=str, help="Experiment to process")
    parser.add_argument(
        "--use_std_err",
        action="store_true",
        help="Use standard error for confidence intervals",
    )
    parser.add_argument(
        "--sgi_sai",
        action="store_true",
        help="Use SGI and SAI closeness values",
    )
    parser.add_argument(
        "--use_cache",
        action="store_true",
        help="Use cached results for processing",
    )
    parser.add_argument(
        "--include-pemirl",
        action="store_true",
        help="Include PEMIRL baseline configs in result collection.",
    )
    args = parser.parse_args()
    INCLUDE_PEMIRL = args.include_pemirl
    if INCLUDE_PEMIRL:
        print("Including PEMIRL baseline configs in process_runs collection.")
    exp_to_process = args.exp_to_process.lower()
    num_runs = 30
    use_cache = args.use_cache
    if use_cache:
        # warn user that cache is being used
        print("### WARNING: Using cached results OR caching results ###")
        print("### RESULTS MAY BE OUTDATED IF {experiment}_results.npz CACHE EXISTS ###")

    using_gt_reach = exp_to_process in ["noise", "adjust", "n", "data"]
    using_gt_pp = exp_to_process in ["pickplace", "ppdata"]
    if using_gt_reach:
        gt_baselines = process_reach_gt_baseline(num_runs)
    elif using_gt_pp:
        gt_baselines = process_pickplace_gt_baseline(num_runs)
    else:
        print(f"Not using GT baselines for {exp_to_process} experiment.")

    if using_gt_reach or using_gt_pp:
        print(f"GT baseline results: {gt_baselines}")

    if exp_to_process == "noise":
        # gt_baselines = process_reach_gt_baseline(num_runs)
        process_noise(gt_baselines, num_runs, use_cache, sgi_sai=args.sgi_sai)
    elif exp_to_process == "adjust":
        # gt_baselines = process_reach_gt_baseline(num_runs)
        process_adjust(gt_baselines, num_runs, use_cache, sgi_sai=args.sgi_sai)
    elif exp_to_process == "n":
        # gt_baselines = process_reach_gt_baseline(num_runs)
        process_n(gt_baselines, num_runs, use_cache, sgi_sai=args.sgi_sai)
    elif exp_to_process == "data":
        # gt_baselines = process_reach_gt_baseline(num_runs)
        process_data(gt_baselines, num_runs, use_cache, sgi_sai=args.sgi_sai)
    elif exp_to_process == "pickplace":
        # gt_baselines = process_pickplace_gt_baseline(num_runs)
        process_pickplace(gt_baselines, num_runs, use_cache, sgi_sai=args.sgi_sai)
    elif exp_to_process == "ppdata":
        # gt_baselines = process_pickplace_gt_baseline(num_runs)
        process_ppdata(gt_baselines, num_runs, use_cache, sgi_sai=args.sgi_sai)
    elif exp_to_process == "tst-ablation":
        process_tst_ablation(use_cache=use_cache)
    elif exp_to_process in {"pemirl-smoke", "pemirl_smoke"}:
        process_pemirl_smoke(num_runs=1, use_cache=use_cache)
    else:
        raise ValueError(f"Unknown experiment: {exp_to_process}")
