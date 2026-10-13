"""Exercise the public launcher without requiring GPUs or Accelerate."""

import os
from pathlib import Path
import shlex
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "train_maskwam.sh"


class TrainLauncherTests(unittest.TestCase):
    def setUp(self):
        self.base_env = {
            "PATH": os.environ["PATH"],
            "MASKWAM_PYTHON": sys.executable,
            "MASKWAM_LAUNCH_DRY_RUN": "1",
        }

    def launch(self, env=None, overrides=(), platform="xtrainer"):
        return subprocess.run(
            ["bash", str(LAUNCHER), platform, *overrides],
            env={**self.base_env, **(env or {})},
            capture_output=True,
            text=True,
            timeout=30,
        )

    @staticmethod
    def distributed_env(rank):
        return {
            "NNODES": "2",
            "GPUS_PER_NODE": "4",
            "NODE_RANK": str(rank),
            "MASTER_ADDR": "10.0.0.10",
        }

    @staticmethod
    def option(command, name):
        return command[command.index(name) + 1]

    def command(self, result):
        self.assertEqual(result.returncode, 0, result.stderr)
        return shlex.split(result.stdout.splitlines()[-1])

    def test_two_hosts_form_one_eight_process_job(self):
        for rank in range(2):
            with self.subTest(rank=rank):
                command = self.command(self.launch(self.distributed_env(rank)))
                self.assertEqual(self.option(command, "--num_processes"), "8")
                self.assertEqual(self.option(command, "--num_machines"), "2")
                self.assertEqual(self.option(command, "--machine_rank"), str(rank))
                self.assertEqual(self.option(command, "--main_process_ip"), "10.0.0.10")
                self.assertIn("--multi_gpu", command)

    def test_hydra_overrides_are_forwarded(self):
        overrides = (
            "training.batch_size=2",
            "exp_name=maskwam_libero_bs2",
        )
        command = self.command(self.launch(overrides=overrides, platform="libero"))
        self.assertEqual(command[-3:], ["dataset=libero", *overrides])

    def test_invalid_machine_count_is_rejected(self):
        result = self.launch({"NNODES": "0"})
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Invalid machine count", result.stderr)

    def test_defaults_to_one_process(self):
        command = self.command(self.launch())
        self.assertEqual(self.option(command, "--num_processes"), "1")
        self.assertEqual(self.option(command, "--num_machines"), "1")
        self.assertEqual(self.option(command, "--machine_rank"), "0")
        self.assertNotIn("--multi_gpu", command)


if __name__ == "__main__":
    unittest.main()
