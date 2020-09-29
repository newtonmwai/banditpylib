from typing import List, Tuple, Set, Optional

import numpy as np

from banditpylib.bandits import search_best_assortment, Reward, \
    local_search_best_assortment
from .utils import OrdinaryMNLLearner


class ThompsonSampling(OrdinaryMNLLearner):
  """Thompson sampling policy :cite:`DBLP:conf/colt/AgrawalAGZ17`"""
  def __init__(self,
               revenues: np.ndarray,
               horizon: int,
               reward: Reward,
               card_limit=np.inf,
               name=None,
               use_local_search=False,
               random_neighbors=10):
    """
    Args:
      revenues: product revenues
      horizon: total number of time steps
      reward: reward the learner wants to maximize
      card_limit: cardinality constraint
      name: alias name
      use_local_search: whether to use local search for searching the best
        assortment
      random_neighbors: number of random neighbors to look up if local search is
        used
    """
    super().__init__(
        revenues=revenues,
        horizon=horizon,
        reward=reward,
        card_limit=card_limit,
        name=name,
        use_local_search=use_local_search,
        random_neighbors=random_neighbors)

  def _name(self) -> str:
    """
    Returns:
      default learner name
    """
    return 'thompson_sampling'

  def reset(self):
    """Learner reset

    Initialization. This function should be called before the start of the game.
    """
    # current time step
    self.__time = 1
    # current episode
    self.__episode = 1
    # number of episodes a product is served until the current episode
    # (exclusive)
    self.__serving_episodes = np.zeros(self.product_num() + 1)
    # number of times a product is picked until the current time (exclusive)
    self.__product_picks = np.zeros(self.product_num() + 1)
    self.__last_actions = None
    self.__last_feedback = None
    # flag to denote whether the initial warm start stage has finished
    self.__done_warm_start = False
    # next product to try in the warm start stage
    self.__next_product_in_warm_start = 1

  def warm_start(self) -> List[Tuple[Set[int], int]]:
    """Initial warm start stage

    Returns:
      assortments to serve in the warm start stage
    """
    # check if last observation is a purchase
    if self.__last_feedback is not None and self.__last_feedback[0][1][0] != 0:
      # continue serving the same assortment
      return self.__last_actions
    self.__last_actions = [({self.__next_product_in_warm_start}, 1)]
    self.__next_product_in_warm_start += 1
    return self.__last_actions

  def within_warm_start(self) -> bool:
    """
    Returns:
      `True` if the learner is still in warm start stage
    """
    return not self.__done_warm_start

  def correlated_sampling(self) -> np.ndarray:
    """
    Returns:
      correlated sampling of preference parameters
    """
    theta = np.max(np.random.normal(0, 1, self.card_limit()))
    # unbiased estimate of preference parameters
    unbiased_est = self.__product_picks / self.__serving_episodes
    sampled_preference_params = unbiased_est + theta * (
        np.sqrt(50 * unbiased_est *
                (unbiased_est + 1) / self.__serving_episodes) +
        75 * np.sqrt(np.log(self.horizon() * self.card_limit())) /
        self.__serving_episodes)
    sampled_preference_params[0] = 1
    sampled_preference_params = np.minimum(sampled_preference_params, 1)
    return sampled_preference_params

  def actions(self, context=None) -> Optional[List[Tuple[Set[int], int]]]:
    """
    Args:
      context: context of the ordinary mnl bandit which should be `None`

    Returns:
      assortments to serve
    """
    del context
    # check if the learner should stop the game
    if self.__time > self.horizon():
      self.__last_actions = None
    # check if still in warm start stage
    elif self.within_warm_start():
      self.__last_actions = self.warm_start()
    else:
      # check if last observation is a purchase
      if self.__last_feedback and self.__last_feedback[0][1][0] != 0:
        # continue serving the same assortment
        return self.__last_actions

      # When a non-purchase observation happens, a new episode is started. Also
      # a new assortment to be served using new estimate of preference
      # parameters is generated.
      # set preference parameters generated by thompson sampling
      self.reward.set_preference_params(self.correlated_sampling())
      # calculate best assortment using the generated preference parameters
      if self.use_local_search:
        # initial assortment to start for local search
        if self.__last_actions is not None:
          init_assortment = self.__last_actions[0][0]
        else:
          init_assortment = None
        _, best_assortment = local_search_best_assortment(
            reward=self.reward,
            random_neighbors=self.random_neighbors,
            card_limit=self.card_limit(),
            init_assortment=init_assortment)
      else:
        _, best_assortment = search_best_assortment(
            reward=self.reward, card_limit=self.card_limit())

      self.__last_actions = [(best_assortment, 1)]
      self.__first_step_after_warm_start = False
    return self.__last_actions

  def update(self, feedback: List[Tuple[np.ndarray, List[int]]]):
    """Learner update

    Args:
      feedback: feedback returned by the ordinary bandit by executing
        `self.__last_actions`.
    """
    self.__product_picks[feedback[0][1][0]] += 1
    # no purchase is observed
    if feedback[0][1][0] == 0:
      for product_id in self.__last_actions[0][0]:
        self.__serving_episodes[product_id] += 1
      # check if it is the end of initial warm start stage
      if not self.__done_warm_start and \
          self.__next_product_in_warm_start > self.product_num():
        self.__done_warm_start = True
        self.__last_actions = None
      self.__episode += 1
    self.__last_feedback = feedback
    self.__time += 1
