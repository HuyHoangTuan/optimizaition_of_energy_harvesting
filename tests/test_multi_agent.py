import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

from src.multi_agent.agent import AgentConfig, IndependentDQN
from src.multi_agent.environment import EnvironmentConfig, JointEnvironment, dbm
from src.multi_agent.trainer import MultiAgentTrainer


def fixed_trace(config, ambient=.1):
    t, n, m = config.slots, config.num_su, config.num_pu
    return {"rt": np.ones((t, n, n)), "sp": np.zeros((t, n, m)),
            "pr": np.zeros((t, n, m)), "ps": np.zeros((t, n, m)),
            "primary_power": np.ones((t, n, m)) * .1, "ambient": np.full(t, ambient)}


class EnvironmentTests(unittest.TestCase):
    def test_primary_count_contract(self):
        for count in (1, 3):
            with self.assertRaises(ValueError):
                EnvironmentConfig(num_pu=count)

    def test_exact_revision2_fixture_and_reset(self):
        c = EnvironmentConfig(num_su=1, slots=4, capacity=.2)
        env = JointEnvironment(c, fixed_trace(c), actions=((1, .125, .4), (0, .125, .4), (0, 1, .4)))
        np.testing.assert_array_equal(env.reset(), np.zeros((1, env.observation_size)))
        for index, battery, energy, executed, debit in ((0, 0, 0, False, 0), (1, .1, .1, True, .075),
                                                        (2, .125, .1, False, 0), (0, .2, .1, False, 0)):
            obs, _, _, info = env.step([index])
            self.assertAlmostEqual(obs[0, 1], energy)
            self.assertAlmostEqual(obs[0, 2], battery)
            self.assertEqual(info["executed"], [executed])
            self.assertAlmostEqual(info["debit"][0], debit)
        with self.assertRaises(RuntimeError):
            env.step([0])
        self.assertEqual(len(env.records), 4)
        self.assertFalse(env.reset().any())
        self.assertEqual(env.records, [])
        self.assertEqual(env.time, 0)
        self.assertEqual(env.step([1])[3]["executed"], [False])

    def test_heterogeneous_interference_execution_and_primary(self):
        c = EnvironmentConfig(num_su=3, slots=5, capacity=2)
        trace = fixed_trace(c, ambient=1)
        trace["rt"][:] = [[1, 2, 3], [5, 7, 11], [13, 17, 19]]
        env = JointEnvironment(c, trace, actions=((1, .125, .4), (0, .125, .4), (0, .25, .75), (0, 3, .1)))
        env.reset()
        env.step([0, 0, 0])
        _, rewards, _, info = env.step([1, 2, 0])
        expected = [2 * .25 * dbm(.25), 5 * .6 * dbm(.125), 13 * .6 * dbm(.125) + 17 * .25 * dbm(.25)]
        np.testing.assert_allclose(info["interference"], expected)
        self.assertEqual(info["executed"], [True, True, False])
        self.assertAlmostEqual(rewards[0], np.log2(1 + .6 * dbm(.125) / (1 + expected[0])))
        _, _, _, info = env.step([1, 3, 0])
        self.assertEqual(info["executed"], [True, False, False])
        self.assertEqual(info["interference"][0], 0)
        # Primary constraint rejection also excludes an otherwise affordable transmitter.
        trace["sp"][:, 1, :] = 100
        env = JointEnvironment(c, trace, actions=((1, .125, .4), (0, .125, .4)))
        env.reset()
        env.step([0, 0, 0])
        _, _, _, info = env.step([1, 1, 0])
        self.assertEqual(info["executed"], [True, False, False])
        self.assertEqual(info["interference"][0], 0)

    def test_invalid_vectors_are_atomic_including_rng(self):
        env = JointEnvironment(EnvironmentConfig(num_su=2, slots=3))
        env.reset()
        state = copy.deepcopy(env.rng.bit_generator.state)
        before = (env.time, env.battery.copy(), env.harvest.copy(), env.debit.copy(), copy.deepcopy(env.records))
        for invalid in (None, 0, [], [0], [0, 0, 0], [0, -1], [0, 10000], [0, .0], [0, True], [0, "1"]):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                env.step(invalid)
            self.assertEqual(env.time, before[0])
            np.testing.assert_array_equal(env.battery, before[1])
            np.testing.assert_array_equal(env.harvest, before[2])
            np.testing.assert_array_equal(env.debit, before[3])
            self.assertEqual(env.records, before[4])
            self.assertEqual(env.rng.bit_generator.state, state)

    def test_permutation_equivalence_and_local_observation(self):
        c = EnvironmentConfig(num_su=3, slots=3, capacity=2)
        trace = fixed_trace(c, ambient=1)
        trace["rt"][:] = [[1, 2, 3], [5, 7, 11], [13, 17, 19]]
        trace["ps"][:] = np.arange(6).reshape(3, 2) / 10
        permutation = [2, 0, 1]
        permuted = {}
        for key, a in trace.items():
            if key == "rt":
                permuted[key] = a[:, permutation, :][:, :, permutation]
            elif key == "ambient":
                permuted[key] = a
            else:
                permuted[key] = a[:, permutation, :]
        actions = ((1, .125, .4), (0, .125, .4), (0, .25, .75))
        left, right = JointEnvironment(c, trace, actions), JointEnvironment(c, permuted, actions)
        left.reset()
        right.reset()
        for joint in ([0, 0, 0], [1, 2, 0], [2, 0, 1]):
            obs, rewards, done, info = left.step(joint)
            obs2, rewards2, done2, info2 = right.step([joint[i] for i in permutation])
            np.testing.assert_allclose(rewards2, rewards[permutation])
            np.testing.assert_allclose(info2["battery"], np.array(info["battery"])[permutation])
            expected = obs[permutation].copy()
            expected[:, 3:6] = expected[:, 3:6][:, permutation]
            np.testing.assert_allclose(obs2, expected)
            self.assertEqual(done, done2)
        self.assertEqual(obs.shape, (3, 3 + 3 + 3 * 2))
        np.testing.assert_array_equal(obs[0, 3:6], trace["rt"][2, 0])
        # Other SU battery changes do not appear in SU0's observation block.
        isolated = JointEnvironment(c, trace, actions)
        isolated.reset()
        isolated.step([0, 0, 0])
        isolated.battery[1] = .123
        own, _, _, _ = isolated.step([0, 0, 0])
        control = JointEnvironment(c, trace, actions)
        control.reset()
        control.step([0, 0, 0])
        normal, _, _, _ = control.step([0, 0, 0])
        np.testing.assert_array_equal(own[0], normal[0])

    def test_legacy_harvest_branches_and_primary_schedule(self):
        c = EnvironmentConfig(num_su=1, slots=3, primary_switch=1)
        trace = fixed_trace(c, ambient=.1)
        env = JointEnvironment(c, trace, actions=((1, .125, .4), (0, 1, .4)))
        env.reset()
        obs, rewards, _, _ = env.step([0])
        self.assertEqual(obs[0, 0], 1)
        self.assertEqual(rewards[0], 0)
        obs, rewards, _, _ = env.step([0])
        self.assertEqual(obs[0, 0], 0)
        self.assertEqual(rewards[0], -1)
        self.assertEqual(env.step([1])[1][0], -1)


class AgentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_independent_actual_update_terminal_only_and_detached_target(self):
        config = AgentConfig(hidden=8, batch_size=2, target_interval=10)
        a, b = [IndependentDQN(i, 4, 3, 100 + i, config) for i in range(2)]
        self.assertIsNot(a.optimizer, b.optimizer)
        self.assertIsNot(a.replay, b.replay)
        self.assertIsNot(a.rng, b.rng)
        for p, q in zip(a.policy.parameters(), b.policy.parameters()):
            self.assertNotEqual(p.data_ptr(), q.data_ptr())
        before_a = [p.detach().clone() for p in a.policy.parameters()]
        before_b = [p.detach().clone() for p in b.policy.parameters()]
        before_target = [p.detach().clone() for p in a.target.parameters()]
        for _ in range(2):
            a.remember(np.ones(4), 0, 10, np.zeros(4), True)
        loss = a.optimize()
        self.assertTrue(np.isfinite(loss))
        self.assertTrue(any(not torch.equal(p, q) for p, q in zip(before_a, a.policy.parameters())))
        self.assertTrue(all(torch.equal(p, q) for p, q in zip(before_b, b.policy.parameters())))
        self.assertTrue(all(torch.equal(p, q) for p, q in zip(before_target, a.target.parameters())))
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in a.target.parameters()))
        self.assertFalse(a.target.training)
        self.assertEqual(b.updates, 0)
        self.assertTrue(a.optimizer.state)
        self.assertFalse(b.optimizer.state)
        self.assertEqual(len(b.replay), 0)

    def test_local_reward_replay_and_seeded_repeatability(self):
        ec = EnvironmentConfig(num_su=2, slots=5, seed=13)
        ac = AgentConfig(hidden=8, batch_size=2)
        trainers = [MultiAgentTrainer(ec, ac), MultiAgentTrainer(ec, ac)]
        initial = [[p.detach().clone() for p in a.policy.parameters()] for a in trainers[0].agents]
        first, second = [trainer.train_episode() for trainer in trainers]
        self.assertEqual(first, second)
        self.assertEqual(len(trainers[0].agents), 2)
        for i, agent in enumerate(trainers[0].agents):
            self.assertGreater(agent.updates, 0)
            self.assertTrue(any(not torch.equal(p, q) for p, q in zip(initial[i], agent.policy.parameters())))
            for transition, record in zip(agent.replay, first):
                self.assertEqual(transition[2], record["rewards"][i])
                np.testing.assert_allclose(transition[0], record["observations"][i])
                self.assertEqual(transition[1], record["action_indices"][i])
                self.assertEqual(transition[4], record["terminal"])
            for p, q in zip(agent.policy.parameters(), trainers[1].agents[i].policy.parameters()):
                self.assertTrue(torch.equal(p, q))
        self.assertNotEqual(trainers[0].agents[0].seed, trainers[0].agents[1].seed)
        self.assertTrue(any(r["rewards"][0] != r["rewards"][1] for r in first))

    def test_artifacts_checkpoint_identity_and_nonoverwrite(self):
        trainer = MultiAgentTrainer(EnvironmentConfig(num_su=2, slots=4), AgentConfig(hidden=8, batch_size=2))
        with tempfile.TemporaryDirectory() as folder:
            output, summary = trainer.run(1, Path(folder) / "fresh")
            self.assertTrue(all(n == 3 for n in summary["updates"]))
            metadata = json.loads((output / "metadata.json").read_text())
            self.assertEqual(metadata["reward_scope"], "own_su")
            self.assertEqual(len(metadata["agent_seeds"]), 2)
            self.assertIn("src/multi_agent/environment.py", metadata["source_sha256"])
            records = [json.loads(line) for line in (output / "transitions.jsonl").read_text().splitlines()]
            self.assertEqual(len(records), 4)
            self.assertTrue(records[-1]["terminal"])
            for i in range(2):
                checkpoint = torch.load(output / ("su_%03d.pt" % i), map_location="cpu", weights_only=True)
                self.assertEqual(checkpoint["su_id"], i)
                self.assertEqual(checkpoint["seed"], trainer.agents[i].seed)
                self.assertEqual(checkpoint["updates"], 3)
                for name, value in trainer.agents[i].policy.state_dict().items():
                    self.assertTrue(torch.equal(checkpoint["policy"][name], value))
            before = {p.name: p.read_bytes() for p in output.iterdir()}
            with self.assertRaises(FileExistsError):
                trainer.run(1, output)
            self.assertEqual(before, {p.name: p.read_bytes() for p in output.iterdir()})
            with self.assertRaises(FileExistsError):
                trainer.run(1, folder)

    def test_fresh_import_has_no_side_effects_or_legacy_modules(self):
        project = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as folder:
            env = os.environ.copy()
            env["PYTHONDONTWRITEBYTECODE"] = "1"
            env["PYTHONPATH"] = str(project) + os.pathsep + env.get("PYTHONPATH", "")
            script = "import sys; import src.multi_agent.environment, src.multi_agent.agent, src.multi_agent.trainer, src.multi_agent.cli; assert not any(m.startswith(('src.modules', 'src.utils')) for m in sys.modules)"
            result = subprocess.run([sys.executable, "-c", script], cwd=folder, env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(list(Path(folder).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
