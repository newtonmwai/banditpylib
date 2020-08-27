import numpy as np
import pytest

from .ordinary_mnl_bandit import OrdinaryMNLBandit, \
    search, search_best_assortment, MeanReward, CvarReward, \
    local_search_best_assortment


class TestOrdinaryMNLBandit:
  """Tests in ordinary mnl bandit"""

  def test_search_unrestricted(self):
    results = []
    search(results, 3, 1, [])
    assert results == [[1, 2, 3], [1, 2], [1, 3], [1], [2, 3], [2], [3]]

  def test_search_restricted(self):
    results = []
    search(results, 3, 1, [], 1)
    assert results == [[1], [2], [3]]

  def test_search_best_assortment(self):
    reward = MeanReward()
    reward.set_abstraction_params(np.array([1, 1, 1, 1, 1]))
    reward.set_revenues(np.array([0, 0.45, 0.8, 0.9, 1.0]))
    best_revenue, best_assortment = search_best_assortment(reward=reward)
    assert best_assortment == [2, 3, 4]
    assert best_revenue == pytest.approx(0.675, 1e-8)

    reward = CvarReward(0.7)
    reward.set_abstraction_params(np.array([1, 0.7, 0.8, 0.5, 0.2]))
    reward.set_revenues(np.array([0, 0.7, 0.8, 0.9, 1.0]))
    best_revenue, best_assortment = search_best_assortment(reward=reward)
    assert best_assortment == [1, 2, 3, 4]
    assert best_revenue == pytest.approx(0.39, 1e-2)


  def test_local_search_best_assortment(self):
    reward = CvarReward(0.7)
    reward.set_abstraction_params(
        np.array([
            1, 0.41796638065213765, 0.640233815546399, 0.8692331344533714,
            0.6766260654759216, 0.7388897283940284, 0.6995225221701101,
            0.44577179791896593, 0.12871019989950505, 0.9006827155203128,
            0.9638849244678077
        ]))
    reward.set_revenues(
        np.array([
            0, 0.6741252008863028, 0.16873112382668315, 0.3056694804088398,
            0.9369761261650644, 0.9444412097177997, 0.6459778680040282,
            0.1391433754809273, 0.7529605722204439, 0.24240120158588363,
            0.9428414848973209
        ]))
    product_num = len(reward.revenues) - 1
    card_limit = 4
    best_revenue, best_assortment = local_search_best_assortment(
        reward=reward,
        search_times=10,
        card_limit=card_limit)
    assert len(best_assortment) <= card_limit
    assert set(best_assortment).issubset(set(range(1, product_num + 1)))
    assert best_revenue > 0


  def test_cvar_calculation(self):
    reward = CvarReward(alpha=0.5)
    reward.set_abstraction_params(np.array([1, 1, 1, 1]))
    reward.set_revenues(np.array([0, 1, 1, 1]))
    # calculate cvar under percentile 0.25
    cvar_alpha = reward.calc([2, 3])
    # upper bound of cvar is 1
    assert cvar_alpha <= 1

    # equivalent to MeanReward
    reward = CvarReward(alpha=2.0)
    reward.set_abstraction_params(np.array([1, 1, 1, 1]))
    reward.set_revenues(np.array([0, 1, 1, 1]))
    # calculate cvar under percentile 0.25
    cvar_alpha = reward.calc([1, 2, 3])
    # upper bound of cvar is 1
    assert cvar_alpha == 0.75

  def test_regret(self):
    abstraction_params = np.array(
        [1.0, 1.0, 1.0, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.5, 0.5])
    revenues = np.array([0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    card_limit = 1
    bandit = OrdinaryMNLBandit(abstraction_params, revenues, card_limit)
    bandit.reset()
    # serve best assortment [1] for 3 times
    bandit.feed([([1], 3)])
    assert bandit.regret() == 0.0

  def test_one_product(self):
    abstraction_params = [1.0, 0.0]
    revenues = [0.0, 1.0]
    bandit = OrdinaryMNLBandit(abstraction_params, revenues)
    bandit.reset()
    # always get no purchase
    assert set(bandit.feed([([1], 5)])[0][1]) == {0}
