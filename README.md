# Supervised Reward Inference

Code for the RLC 2026 paper **"Supervised Reward Inference"** by Will Schwarzer, Jordan Schneider, Philip S. Thomas, and Scott Niekum.

Paper: [OpenReview](https://openreview.net/forum?id=3P2GJSvuIJ) · [arXiv](https://arxiv.org/abs/2502.18447)

SRI infers reward functions from arbitrarily suboptimal or communicative behavior by treating reward inference as supervised learning: given a dataset of behavior paired with known rewards, it learns the mapping from behavior to rewards directly. The paper shows that SRI is asymptotically Bayes-optimal under standard assumptions, achieves near-ceiling performance on a prior tabular benchmark for reward inference from suboptimal behavior, and infers accurate rewards from highly suboptimal demonstrations on Meta-World robotics tasks. The same framework generalizes directly to goal prediction (SGI) and action prediction (SAI).

## Repository layout

| Path | Contents |
| --- | --- |
| `sri/` | Main SRI code. `run_experiment.py` (training/inference/RL orchestrator), `process_runs.py` (figure and table generation), `reward_inference/` (reward-model training), `rl/`, `pemirl/` (PEMIRL baseline) |
| `config/` | Composable YAML configs (`datasets/`, `train/`, `inference/`, `rl/`, `model/`, `baselines/`, `general/`) mixed by `run_experiment.py` |
| `scripts/experiments/`, `scripts/rl/` | Slurm scripts used to run the paper's Meta-World experiments and generate data |
| `learning_biases/`, `gridworld/`, `sri/learning_biases_bridge/`, `scripts/learning_biases_bridge/` | Code for the tabular benchmark experiments (Figure 4), building on Shah et al.'s [learning_biases](https://github.com/HumanCompatibleAI/learning_biases) |
| `imitation/`, `metaworld/`, `set_transformer/`, `stable-baselines3/` | Vendored dependencies (see [Third-party code](#third-party-code)) |

## Installation

Two conda environments are used: a main environment for all SRI/Meta-World code, and a separate TF1 environment only needed for the tabular-benchmark (Figure 4) baselines.

On Linux (the platform the experiments ran on), use the exact exported environment:

```bash
# Main environment
conda env create -f env.yml
conda activate meta-world

# TF1 environment (only for learning_biases/ baselines)
conda env create -f env_learning_biases_tf1.yml
```

On macOS or Windows (WSL recommended), use the portable spec instead — same package pins, without the Linux-specific conda packages:

```bash
conda env create -f env_portable.yml
conda activate sri
```

Rendering defaults to headless `osmesa`; on a machine with a display, set `MUJOCO_GL` accordingly (e.g. `MUJOCO_GL=glfw` on macOS). The TF1 environment is Linux-only (TensorFlow 1.x has no Apple-Silicon builds), but it is needed only for the Figure 4 baseline comparisons.

Then expose the vendored packages, from the repository root:

```bash
export PYTHONPATH="$PWD:$PWD/stable-baselines3:$PWD/imitation/src:$PWD/set_transformer:$PWD/learning_biases:$PYTHONPATH"
```

(Editable installs of the vendored packages work too.)

Finally, fetch the Meta-World MuJoCo assets (~38M of meshes and textures, not shipped in this repo to keep clones small):

```bash
bash scripts/fetch_metaworld_assets.sh
```

This pulls the V2 assets from upstream [Metaworld](https://github.com/Farama-Foundation/Metaworld) at a pinned commit; the two asset files modified in our fork ship with this repo and are left untouched.

## Running experiments

The Meta-World pipeline has three stages, all driven by `sri.run_experiment` with composed configs:

1. **Data generation** — scripted policies produce demonstration/reward datasets (`scripts/rl/`).
2. **Reward model training and inference** — train the behavior-conditioned reward model (`--train-config`), then condition on demonstrations to infer rewards.
3. **Policy optimization and evaluation** — train RL (TQC/PPO) on inferred rewards and evaluate on ground truth.

A typical invocation (from the paper's noise-robustness experiment):

```bash
python -m sri.run_experiment \
    --general-config general/default_experiment.yml \
    --train-config train/skip.yml \
    --dataset-config datasets/reach_noise.yml \
    --obs-dataset-config datasets/reach_for_pickplace.yml \
    --model-config model/load_default.yml \
    --rl-config rl/tqc_quarter_legacy_paper_esr_5m.yml \
    --inference-config inference/default_only_rl.yml \
    --dataset-args "noise_coeff=0.5"
```

The scripts in `scripts/experiments/` are the source of truth for the exact config combinations used in the paper; they are Slurm array scripts, so batch-job boilerplate (`#SBATCH -A YOUR_ACCOUNT`, log paths, partitions) needs to be adapted to your cluster, or the inner `python -m sri.run_experiment` commands can be run directly.

Figures and tables are generated from completed runs with `sri.process_runs`:

```bash
python -m sri.process_runs noise --use_cache
```

### Paper figure/table → experiment map

| Paper asset | Experiment key | Main output |
| --- | --- | --- |
| Fig. reach robustness, noisy panel | `noise` | `results/plots/noise_plot_ci.pdf` |
| Fig. reach robustness, offset-goal panel | `adjust` | `results/plots/adjustment_plot_ci.pdf` |
| Fig. demonstration efficiency | `n` | `results/plots/n_experiment_plot_ci.pdf` |
| Table: pick-place from gestures | `pickplace` | `results/plots/pickplace_experiment_plot_ci.pdf` |
| Table: data efficiency (reach / pick-place) | `data` / `ppdata` | `results/tables/*_experiment_plot_ci.txt` |
| Appendix: SRI vs. SGI vs. SAI variants | any key + `--sgi_sai` | as above |
| Appendix: architecture ablation | `tst-ablation` | `results/plots/tst_transformer_ablation_n_plot.pdf` |

### Tabular benchmark (Figure 4)

The comparison on Shah et al.'s suboptimal-planner benchmark lives in the `learning_biases/` path. Main entry points:

```bash
# TF1 environment:
python learning_biases/run_benchmarks.py --help
python learning_biases/bridge_export_dataset.py --help
python learning_biases/bridge_run_shah_given_rewards.py --help

# Main environment:
python -m sri.learning_biases_bridge.train_sri_policy --help
python -m sri.learning_biases_bridge.export_fig4_four_method_tables --help
python learning_biases/create_graphs.py --help
```

Slurm drivers are in `scripts/learning_biases_bridge/`.

## Caveats

- Experiment scripts assume a Slurm cluster and Weights & Biases logging; account, email, and W&B entity/project fields are placeholders (`YOUR_ACCOUNT`, `your_wandb_entity`, ...) that you should replace (see e.g. `config/general/default.yml`). Later pipeline stages locate earlier runs through W&B, so a W&B account is effectively required for full reproduction.
- The vendored `metaworld/` supports the V2 environments only (all paper experiments use V2 goal-observable environments); assets are fetched by `scripts/fetch_metaworld_assets.sh`.
- This is research code, organized around reproducing the paper's experiments rather than as a general-purpose library.

## Third-party code

Vendored dependencies retain their upstream licenses:

- [stable-baselines3](https://github.com/DLR-RM/stable-baselines3) (MIT, license included)
- [imitation](https://github.com/HumanCompatibleAI/imitation) (MIT, license included)
- [Metaworld](https://github.com/Farama-Foundation/Metaworld) (MIT)
- [set_transformer](https://github.com/juho-lee/set_transformer) (MIT)
- [learning_biases](https://github.com/HumanCompatibleAI/learning_biases) (Shah et al.)

## Citation

```bibtex
@inproceedings{schwarzer2026supervised,
  title     = {Supervised Reward Inference},
  author    = {Schwarzer, Will and Schneider, Jordan and Thomas, Philip S. and Niekum, Scott},
  booktitle = {Reinforcement Learning Conference (RLC)},
  year      = {2026}
}
```

## License

MIT (see [LICENSE](LICENSE)); vendored third-party code under its own licenses as noted above.
