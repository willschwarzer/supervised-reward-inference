import importlib.util
import os
import shutil
import subprocess
import tempfile
import unittest


def _has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


@unittest.skipUnless(_has_module("torch") and _has_module("tensorflow"), "Smoke test requires both torch and tensorflow")
class TestBridgeSmoke(unittest.TestCase):
    def test_smoke_pipeline_small(self):
        root = tempfile.mkdtemp(prefix="lb_bridge_smoke_")
        try:
            dataset_dir = os.path.join(root, "run")
            subprocess.check_call(
                [
                    "python",
                    "-m",
                    "learning_biases.bridge_export_dataset",
                    "--agent",
                    "optimal",
                    "--seed",
                    "0",
                    "--planner-train-size",
                    "40",
                    "--planner-val-size",
                    "20",
                    "--reward-infer-size",
                    "20",
                    "--out",
                    dataset_dir,
                ]
            )
            subprocess.check_call(
                [
                    "python",
                    "-m",
                    "learning_biases.bridge_run_shah_given_rewards",
                    "--dataset",
                    dataset_dir,
                    "--out",
                    dataset_dir,
                    "--seed",
                    "0",
                    "--batchsize",
                    "20",
                ]
            )
            subprocess.check_call(
                [
                    "python",
                    "-m",
                    "sri.learning_biases_bridge.train_sri_policy",
                    "--dataset",
                    dataset_dir,
                    "--out",
                    dataset_dir,
                    "--seed",
                    "0",
                    "--epochs",
                    "2",
                ]
            )
            subprocess.check_call(
                [
                    "python",
                    "-m",
                    "sri.learning_biases_bridge.evaluate_sri_policy",
                    "--dataset",
                    dataset_dir,
                    "--pred",
                    os.path.join(dataset_dir, "sri_pred_reward_vec.npy"),
                    "--out",
                    dataset_dir,
                ]
            )
        finally:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
