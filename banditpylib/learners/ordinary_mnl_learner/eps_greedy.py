from typing import List, Tuple

import numpy as np

from banditpylib.bandits import search_best_assortment, Reward, search
from .utils import OrdinaryMNLLearner


class EpsGreedy(OrdinaryMNLLearner):
  r"""Epsilon-Greedy policy

  With probability :math:`\frac{\epsilon}{t}` do uniform sampling and with the
  remaining probability serve the assortment with the maximum empirical reward.
  """
  def __init__(self,
               revenues: np.ndarray,
               horizon: int,
               reward: Reward,
               card_limit=np.inf,
               name=None,
               eps=1.0):
    """
    Args:
      revenues: product revenues
      horizon: total number of time steps
      reward: reward the learner wants to maximize
      card_limit: cardinality constraint
      name: alias name
      eps: epsilon
    """
    super().__init__(revenues, horizon, reward, card_limit, name)
    if eps <= 0:
      raise Exception('Epsilon %.2f in %s is no greater than 0!' % \
          (eps, self.__name))
    self.__eps = eps

  def _name(self):
    return 'epsilon_greedy'

  def reset(self):
    # current time step
    self.__time = 1
    # current episode
    self.__episode = 1
    # number of episodes a product is served until the current episode
    # (exclusive)
    self.__serving_episodes = np.zeros(self.product_num() + 1)
    # number of times the customer chooses a product until the current time
    # (exclusive)
    self.__customer_choices = np.zeros(self.product_num() + 1)
    self.__last_actions = None
    self.__last_feedback = None

  def em_preference_params(self) -> np.ndarray:
    """
    Returns:
      empirical estimate of preference parameters
    """
    # unbiased estimate of preference parameters
    unbiased_est = self.__customer_choices / self.__serving_episodes
    unbiased_est[np.isnan(unbiased_est)] = 1
    unbiased_est = np.minimum(unbiased_est, 1)
    return unbiased_est

  def select_ramdom_assort(self) -> List[int]:
    assortments = []
    search(assortments, self.product_num(), 1, [], self.card_limit())
    # pylint: disable=E1101
    return assortments[int(np.random.randint(0, len(assortments)))]

  def actions(self, context=None) -> List[Tuple[List[int], int]]:
    """
    Returns:
      assortments to serve
    """
    del context
    if self.__time > self.horizon():
      self.__last_actions = None
    else:
      # check if last observation is not a non-purchase
      if self.__last_feedback and self.__last_feedback[0][1][0] != 0:
        return self.__last_actions
      # When a non-purchase observation happens, a new episode is started and
      # a new assortment to be served is calculated

      # pylint: disable=E1101
      # with probability eps/t, randomly select an assortment to serve
      if np.random.random() <= self.__eps / self.__time:
        self.__last_actions = [(self.select_ramdom_assort(), 1)]
        return self.__last_actions

      self.reward.set_preference_params(self.em_preference_params())
      # calculate assortment with the maximum reward using optimistic
      # preference parameters
      _, best_assortment = search_best_assortment(
          reward=self.reward,
          card_limit=self.card_limit())
      self.__last_actions = [(best_assortment, 1)]
    return self.__last_actions

  def update(self, feedback):
    self.__customer_choices[feedback[0][1][0]] += 1
    self.__last_feedback = feedback
    self.__time += 1
    if feedback[0][1][0] == 0:
      for product_id in self.__last_actions[0][0]:
        self.__serving_episodes[product_id] += 1
      self.__episode += 1