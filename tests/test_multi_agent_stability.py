import copy
import unittest
import numpy as np
import torch

from src.multi_agent.agent import AgentConfig, IndependentDQN
from src.multi_agent.environment import EnvironmentConfig, JointEnvironment
from src.multi_agent.stability import (evaluate, select_candidate, training_fingerprint, q_bounds,
    ensure_audit_ready, ensure_training_allowed, screen_promotion)
from test_multi_agent import fixed_trace


class StabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_double_dqn_disagreement_terminal_and_detachment(self):
        batch = [(np.zeros(2,dtype=np.float32), 0, 2., np.zeros(2,dtype=np.float32), False),
                 (np.zeros(2,dtype=np.float32), 0, 7., np.zeros(2,dtype=np.float32), True)]
        for double, expected in ((False, 2+.99*9), (True, 2+.99*3)):
            a = IndependentDQN(0, 2, 2, 2, AgentConfig.legacy_parity(batch_size=2, double_dqn=double))
            with torch.no_grad():
                for p in a.policy.parameters(): p.zero_()
                for p in a.target.parameters(): p.zero_()
                a.policy[-1].bias.copy_(torch.tensor([4.,1.]))
                a.target[-1].bias.copy_(torch.tensor([3.,9.]))
            loss = a.optimize_batch(batch)
            self.assertAlmostEqual(a.last_diagnostics["target_mean"], (expected+7)/2, places=5)
            self.assertAlmostEqual(loss, ((4-expected)**2+(4-7)**2)/2, places=4)
            self.assertTrue(all(p.grad is None for p in a.target.parameters()))
        a.optimize_batch([(*batch[0][0:4], True), batch[1]])
        self.assertEqual(a.last_diagnostics["target_mean"], 4.5)

    def test_episode_schedule_zero_and_five(self):
        a = IndependentDQN(0, 2, 2, 2, AgentConfig.legacy_parity())
        original = {k:v.clone() for k,v in a.target.state_dict().items()}
        with torch.no_grad():
            for p in a.policy.parameters(): p.add_(1)
        a.end_episode(0)
        for k,v in a.target.state_dict().items():
            torch.testing.assert_close(v, a.policy.state_dict()[k]*.001+original[k]*.999, rtol=0, atol=0)
        intermediate = copy.deepcopy(a.target.state_dict())
        for index in range(1,5): a.end_episode(index)
        for k,v in a.target.state_dict().items(): self.assertTrue(torch.equal(v,intermediate[k]))
        a.end_episode(5)
        self.assertTrue(any(not torch.equal(v,intermediate[k]) for k,v in a.target.state_dict().items()))

    def test_requested_rejected_debit_and_bank_exhaustion(self):
        actions = ((0,1,.4),(1,.125,.4))
        for mode, expected in (("executed",.2),("requested",0)):
            c=EnvironmentConfig(num_su=1,slots=3,debit_mode=mode)
            e=JointEnvironment(c,actions=actions,trace_bank=[fixed_trace(c)])
            e.reset(); e.step([1]); e.step([0]); obs,*_=e.step([1])
            self.assertAlmostEqual(obs[0,2],expected)
            with self.assertRaises(IndexError):e.reset()

    def test_populated_state_frozen_and_invalid_configs(self):
        c=EnvironmentConfig(num_su=1,slots=3)
        a=IndependentDQN(0,10,2,2,AgentConfig(batch_size=2))
        for _ in range(2):a.remember(np.zeros(10),0,1,np.ones(10),False)
        a.optimize();a.select_action(np.zeros(10));a.end_episode()
        before=training_fingerprint([a])
        evaluate([a],c,((0,.125,.4),(1,.125,.4)),[fixed_trace(c)],99)
        self.assertEqual(before,training_fingerprint([a]))
        for options in (dict(optimizer="x"),dict(loss="x"),dict(gradient_clip=-1),dict(target_tau=0),dict(forced_random_episodes=-1)):
            with self.assertRaises(ValueError):AgentConfig(**options)

    def test_selection_win_tie_regression_and_health(self):
        def v(reward, healthy=True):return dict(reward=reward,healthy=healthy)
        anchor=v([1,1,1]); plain=v([1.1,1.1,1.1]); double=v([1.2,1.2,1.2])
        self.assertEqual(select_candidate(anchor,plain,double)["selected"],"double")
        self.assertEqual(select_candidate(anchor,anchor,anchor)["selected"],"anchor")
        self.assertEqual(select_candidate(anchor,v([0,0,0]),v([.2,.2,.2]))["selected"],"anchor")
        self.assertEqual(select_candidate(anchor,v([5,5,5],False),v([8,8,8],False))["selected"],"anchor")
        self.assertEqual(select_candidate(v([1,1,1],False),plain,v([1,1,1],False))["selected"],"plain")
        self.assertIsNone(select_candidate(v([1]*3,False),v([1]*3,False),v([1]*3,False))["selected"])

    def test_audit_requires_durable_completion_and_no_unfinished_cells(self):
        log=dict(locked=False,cells=[dict(status="complete")],training_seconds=1.)
        with self.assertRaises(RuntimeError):ensure_audit_ready(log,None)
        for status in ("reserved","running","partial_budget_stop"):
            log["cells"][0]["status"]=status
            with self.assertRaises(RuntimeError):ensure_audit_ready(log,{"complete":True})
        log["cells"][0]["status"]="complete"
        ensure_audit_ready(log,{"complete":True})
        log["locked"]=True
        with self.assertRaises(RuntimeError):ensure_training_allowed(log)

    def test_shared_sum_reward_bound_units_for_two_three_five_sus(self):
        for n in (2,3,5):
            c=EnvironmentConfig(num_su=n,slots=3)
            actions=((0,.125,.4),(1,.125,.4));bank=[fixed_trace(c)]
            a=IndependentDQN(0,(3+n+6)*n,2,2,AgentConfig(hidden=8,batch_size=2))
            result=evaluate([a],c,actions,bank,99,shared=True)
            np.testing.assert_allclose(result["q_bounds"],np.array(q_bounds(c,actions,bank,.99))*n)
            self.assertEqual(result["q_reward_scope"],"network_sum")

    def test_reward_gain_with_score_collapse_or_zero_execution_rejected(self):
        def row(reward,score,execution):
            return dict(healthy=True,mean_per_episode=dict(reward=[reward],score=[score],executed=[execution]))
        baseline=row(0,1,1)
        for candidate in (row(10,1,2),row(10,0,2),row(10,5,0)):
            result=screen_promotion(candidate,baseline,0,20)
            self.assertGreater(result["gain"],.05)
            self.assertFalse(result["accepted"])
        self.assertTrue(screen_promotion(row(10,5,2),baseline,0,20)["accepted"])


if __name__ == "__main__":unittest.main()
