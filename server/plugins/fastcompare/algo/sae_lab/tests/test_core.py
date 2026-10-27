import importlib.util
from pathlib import Path
import numpy as np
spec = importlib.util.spec_from_file_location('student_sae_core', Path(__file__).resolve().parents[1] / 'core.py')
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)

def test_balanced_counts_and_filter():
    # Neuron 0 has 4 pos and 3 neg, neuron 1 has 3 pos and 3 neg;
    # neuron 2 has no negatives. With group_size=2, order 0 then 1.
    z = np.array([[1,1,1],[2,1,1],[3,1,1],[4,0,1],[-1,-1,1],[-2,-1,1],[-3,-1,1]])
    groups = core.select_groups(z, np.ones(7), group_size=2, seed=123)
    assert [g[0] for g in groups] == [0,1]
    assert [g[3] for g in groups] == [3,3]
    for neuron, pos, neg, score in groups:
        assert len(set(pos)) == len(set(neg)) == 2
        assert all(z[i,neuron] > 0 for i in pos)
        assert all(z[i,neuron] < 0 for i in neg)
    assert groups == core.select_groups(z, np.ones(7), group_size=2, seed=123)

def test_explicit_only():
    z = np.array([[2,3],[-2,4]])
    a = np.zeros(2); b = np.array([5.,5.])
    for response, expected in [('positive',1),('negative',-1),('both',0),('neither',0)]:
        x,y = core.update_preferences(a,b,z,0,[0],[1],response)
        assert x[0] == expected and np.all(y == 0)

def test_recommend_ignores_implicit():
    z=np.array([[1,0],[0,1],[-1,0]],dtype=float)
    assert core.recommend(z,[1,0],[0,100],{0},2)==[1,2]
