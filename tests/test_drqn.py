import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from src.multi_agent.recurrent import IndependentDRQN, RecurrentQNetwork
from src.multi_agent.drqn_evaluation import evaluate, populated_fingerprint, stage_gate
from src.multi_agent.environment import EnvironmentConfig
from src.multi_agent.drqn_traces import (source_bank, save_bank, load_bank, bank_identity,
                                       reject_overlap, checked_legacy_constructor)
from src.multi_agent.drqn_experiment import (Ledger, BudgetStop, forecast, write_new, freeze,
    verify_identity, verify_seal, audit, workload, make_agents, scientific_sources)


def add_episode(agent, offset=0, acting=False):
    for slot in range(20):
        state=np.full(agent.observation_size,offset+slot,dtype=np.float32)
        following=state+1
        action=agent.select_action(state) if acting else slot%agent.num_actions
        agent.remember(state,action,float(slot%3),following,slot==19)
    agent.end_episode()


class CumulativeNetwork(nn.Module):
    """History-sensitive oracle with a trainable scalar."""
    def __init__(self):
        super().__init__();self.weight=nn.Parameter(torch.tensor(1.))
    def forward(self,x,hidden=None):
        q=x.cumsum(1)*self.weight
        return torch.cat((q,q+1),2),None


class RecurrentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_batch_and_temporal_independence(self):
        torch.manual_seed(3);net=RecurrentQNetwork(3,2)
        x=torch.randn(4,21,3)
        full,h=net(x)
        for i in range(4):
            individual,ih=net(x[i:i+1])
            torch.testing.assert_close(full[i:i+1],individual)
            torch.testing.assert_close(h[:,i:i+1],ih)
        first,h=net(x[:,:8]);last,_=net(x[:,8:],h)
        torch.testing.assert_close(full,torch.cat((first,last),1))
        with self.assertRaises(ValueError):net(x[0])

    def test_complete_chronological_replay_and_validation(self):
        a=IndependentDRQN(0,1,2,2)
        a.remember([0],0,0,[1],False)
        self.assertEqual(len(a.replay),0)
        with self.assertRaises(ValueError):a.remember([4],0,0,[5],False)
        with self.assertRaises(ValueError):a.remember([1],0,0,[2],True)
        with self.assertRaises(ValueError):a.end_episode()
        for t in range(1,20):a.remember([t],0,t,[t+1],t==19)
        self.assertEqual(len(a.replay),1)
        np.testing.assert_array_equal(a.replay[0]['observations'][:,0],np.arange(21))
        self.assertEqual(a.replay.maxlen,5000)
        a.end_episode();self.assertEqual(a.pending,[])

    def test_full_prefix_target_alignment_and_terminal_reward(self):
        a=IndependentDRQN(0,1,2,2);add_episode(a,1)
        a.policy=CumulativeNetwork();a.target=CumulativeNetwork()
        predicted,expected=a.td_values(list(a.replay))
        x=torch.arange(1,22,dtype=torch.float32)
        oracle=x.cumsum(0)[1:]+1
        rewards=torch.tensor(a.replay[0]['rewards'])
        wanted=rewards+.99*oracle;wanted[-1]=rewards[-1]
        torch.testing.assert_close(expected[0],wanted)
        self.assertFalse(expected.requires_grad)
        self.assertEqual(float(expected[0,-1]),float(rewards[-1]))
        self.assertNotEqual(float(expected[0,0]),.99*(2+1))

    def test_gradients_reach_early_prefix(self):
        torch.manual_seed(5);net=RecurrentQNetwork(2,2)
        x=torch.randn(1,21,2,requires_grad=True)
        q,_=net(x);q[0,19,0].backward()
        self.assertGreater(float(x.grad[0,0].abs().sum()),0)
        self.assertEqual(float(x.grad[0,20].abs().sum()),0)

    def test_strict_masks_warmup_finite_and_private_rng(self):
        a=IndependentDRQN(0,2,2,2)
        for ep in range(3):add_episode(a,ep*30)
        self.assertIsNone(a.optimize())
        add_episode(a,90)
        state=a.mask_rng.getstate();global_state=torch.random.get_rng_state().clone()
        loss=a.optimize();self.assertTrue(np.isfinite(loss));self.assertEqual(a.updates,1)
        positions=a.last_diagnostics['loss_positions']
        self.assertEqual(len(positions),64);self.assertEqual(len(set(positions)),64)
        self.assertTrue(all(0<=v<80 for v in positions));self.assertNotEqual(state,a.mask_rng.getstate())
        torch.testing.assert_close(global_state,torch.random.get_rng_state())
        self.assertEqual(a.last_diagnostics['processed_observations_per_network'],84)
        self.assertTrue(np.isfinite(a.last_diagnostics['gradient_norm']))
        self.assertTrue(all(p.grad is None for p in a.target.parameters()))

    def test_random_action_memory_reset_and_refresh(self):
        a=IndependentDRQN(0,2,2,2)
        a.select_action([1,2]);a.select_action([3,4])
        self.assertEqual(len(a.acting_prefix),2)
        self.assertIsNone(a.hidden.grad_fn)
        with torch.no_grad():_,expected=a.policy(torch.tensor([[[1.,2.],[3.,4.]]]))
        torch.testing.assert_close(a.hidden,expected)
        for ep in range(4):add_episode(a,ep*30)
        a.select_action([1,2]);a.select_action([3,4]);before=a.hidden.clone()
        a.optimize()
        with torch.no_grad():_,expected=a.policy(torch.tensor([[[1.,2.],[3.,4.]]]))
        torch.testing.assert_close(a.hidden,expected)
        self.assertFalse(torch.equal(before,a.hidden))
        a.end_episode();self.assertIsNone(a.hidden);self.assertEqual(a.acting_prefix,[])

    def test_su_own_reward_and_state_independence(self):
        a,b=IndependentDRQN(0,2,2,2),IndependentDRQN(1,2,2,3)
        before=populated_fingerprint([b])
        for ep in range(4):add_episode(a,ep*30)
        a.optimize();a.select_action([1,2])
        self.assertEqual(before,populated_fingerprint([b]))
        self.assertNotEqual(a.policy.gru.weight_ih_l0.data_ptr(),b.policy.gru.weight_ih_l0.data_ptr())

    def test_populated_frozen_evaluation_and_same_runtime_repetition(self):
        a=IndependentDRQN(0,10,32,2)
        for ep in range(4):add_episode(a,ep*30)
        a.select_action(np.ones(10));a.remember(np.ones(10),0,1,np.ones(10)*2,False);a.optimize()
        before=populated_fingerprint([a]);bank,actions=source_bank(1,920001,2)
        first=evaluate([a],EnvironmentConfig(num_su=1),actions,bank,920003)
        second=evaluate([a],EnvironmentConfig(num_su=1),actions,bank,920003)
        self.assertEqual(before,populated_fingerprint([a]));self.assertEqual(first,second)
        self.assertEqual(first['training_before'],first['training_after'])

    def test_seed_repeat_checkpoint_identity_and_parameters(self):
        a,b=IndependentDRQN(0,10,32,2),IndependentDRQN(0,10,32,2)
        for agent in (a,b):
            for ep in range(4):add_episode(agent,ep*30,acting=True)
            agent.optimize()
        self.assertEqual(populated_fingerprint([a]),populated_fingerprint([b]))
        self.assertIn('not exact resume',a.checkpoint()['purpose'])
        self.assertEqual(sum(p.numel() for p in a.policy.parameters()),16672)
        self.assertEqual(sum(p.numel() for p in make_agents('dqn',1,2)[0].policy.parameters()),22048)

    def test_actual_update_opportunities_start_at_transition80(self):
        a=IndependentDRQN(0,2,2,2)
        first=None
        for step in range(100):
            t=step%20;state=np.full(2,t,dtype=np.float32)
            a.remember(state,0,.1,state+1,t==19)
            value=a.optimize()
            if value is not None and first is None:first=step+1
            if t==19:a.end_episode()
        self.assertEqual(first,80);self.assertEqual(a.updates,21)


class RunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_trace_roundtrip_and_canonical_duplicate_detection(self):
        bank,actions=source_bank(1,920001,2)
        again,again_actions=source_bank(1,920001,2)
        self.assertEqual(bank_identity(bank),bank_identity(again));self.assertEqual(actions,again_actions)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'bank.npz';save_bank(path,bank)
            self.assertEqual(bank_identity(bank),bank_identity(load_bank(path)))
            with self.assertRaises(FileExistsError):save_bank(path,bank)
        with self.assertRaises(ValueError):reject_overlap(dict(a=bank,b=copy.deepcopy(bank)))
        other,_=source_bank(1,930002,2)
        self.assertEqual(len(reject_overlap(dict(a=bank,b=other))),2)

    def test_exact_legacy_constructor_match_without_global_rng_mutation(self):
        import random
        before=(random.getstate(),copy.deepcopy(np.random.get_state()),torch.random.get_rng_state().clone())
        reference,actions=checked_legacy_constructor(Path(__file__).resolve().parents[1],2,920001,3)
        generated,new_actions=source_bank(2,920001,3)
        self.assertEqual(bank_identity(reference),bank_identity(generated))
        self.assertEqual(actions,new_actions)
        self.assertEqual(before[0],random.getstate())
        np.testing.assert_equal(before[1],np.random.get_state())
        torch.testing.assert_close(before[2],torch.random.get_rng_state())

    def test_ledger_failed_and_partial_work_charged_no_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);ledger=Ledger(root,cap=10)
            ledger.start('test','pilot','identity')
            with self.assertRaisesRegex(ValueError,'retained'):
                with ledger.step('test'):raise ValueError('retained')
            self.assertGreater(ledger.total(),0)
            ledger.finish('test','failed')
            self.assertAlmostEqual(Ledger(root).total(),ledger.total())
            with self.assertRaises(FileExistsError):ledger.start('test','pilot','identity')
            write_new(root/'audit-open.json',{})
            with self.assertRaisesRegex(RuntimeError,'prohibited'):ledger.allowed()

    def test_cap_forecast_and_unfinished_attempt(self):
        self.assertTrue(forecast([('dqn',10),('drqn',10)],dict(dqn=1,drqn=2),0,60)['admitted'])
        self.assertFalse(forecast([('dqn',10),('drqn',10)],dict(dqn=1,drqn=2),1,60)['admitted'])
        with tempfile.TemporaryDirectory() as folder:
            ledger=Ledger(folder,cap=0)
            with self.assertRaises(BudgetStop):ledger.allowed()
            ledger=Ledger(folder);ledger.start('unfinished','pilot','id')
            with self.assertRaisesRegex(RuntimeError,'unfinished'):ledger.start('next','pilot','id')

    def test_changed_input_rejection_retained_and_analysis_separate(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);write_new(root/'preflight.json',{})
            write_new(root/'inputs/protocol.json',dict(version=2))
            identity=freeze(root);self.assertEqual(verify_identity(root),identity)
            (root/'inputs/protocol.json').write_text('changed')
            with self.assertRaisesRegex(RuntimeError,'changed'):verify_identity(root)
            self.assertEqual(len(list((root/'rejections').glob('*.json'))),1)
        self.assertFalse(any(p.name=='drqn_analysis.py' for p in scientific_sources()))

    def test_changed_scientific_source_rejected_and_seal_stops_training(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source=root/'fixture.py';source.write_text('original')
            write_new(root/'preflight.json',{});write_new(root/'inputs/protocol.json',{})
            with patch('src.multi_agent.drqn_experiment.PROJECT',root), \
                 patch('src.multi_agent.drqn_experiment.scientific_sources',return_value=[source]):
                freeze(root);source.write_text('changed')
                with self.assertRaisesRegex(RuntimeError,'changed'):verify_identity(root)
            write_new(root/'seal.json',{})
            with patch('src.multi_agent.drqn_experiment.verify_identity',return_value='fixture'):
                with self.assertRaisesRegex(RuntimeError,'prohibited'):
                    workload(root,'dqn',1,2)
            self.assertFalse((root/'attempts').exists())

    def test_missing_seal_blocks_audit_before_bank_generation(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);write_new(root/'preflight.json',{});write_new(root/'inputs/protocol.json',{})
            freeze(root)
            with self.assertRaises(FileNotFoundError):audit(root)
            self.assertFalse((root/'audit-open.json').exists())
            self.assertFalse((root/'audit').exists())

    def test_partial_workload_records_and_checkpoints_survive(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);inputs=root/'inputs';inputs.mkdir()
            bank,actions=source_bank(1,1000007,1)
            save_bank(inputs/'pilot_n1.npz',bank);save_bank(inputs/'monitor_n1.npz',bank)
            write_new(inputs/'actions_pilot_n1.json',actions)
            with patch('src.multi_agent.drqn_experiment.verify_identity',return_value='unit_fixture'), \
                 patch('src.multi_agent.agent.IndependentDQN.optimize',side_effect=ValueError('fixture failure')):
                result=workload(root,'dqn',1,1000007,kind='pilot',episodes=1)
            self.assertEqual(result['status'],'failed');self.assertIn('fixture failure',result['failure'])
            self.assertGreater(Ledger(root).total(),0)
            target=root/'attempts/pilot_dqn_n1_seed1000007'
            self.assertTrue((target/'su0_eppartial.pt').exists())
            self.assertTrue((target/'result.json').exists())
            self.assertEqual(Ledger(root).events()[-1]['status'],'failed')

    def test_gate_rejects_nonfinite_even_when_other_flags_are_true(self):
        def metric(reward,finite=True):
            return dict(healthy=True,finite=finite,mean_per_episode=dict(reward=[reward],score=[reward+1],executed=[1]))
        finals=[metric(3),metric(3),metric(3,False)]
        zero=[metric(0)]*3
        gate=stage_gate(finals,zero,zero)
        self.assertFalse(gate['passed']);self.assertFalse(gate['checks']['finite'])


if __name__=='__main__':unittest.main()
