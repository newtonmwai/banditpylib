import numpy as np

from .utils import OrdinaryLearner

__all__ = ['EpsGreedy']


class EpsGreedy(OrdinaryLearner):
  """Epsilon-Greedy Algorithm
  With probability eps/t do uniform sampling and with the left probability,
  pull arm with the maximum empirical mean.
  """

  def __init__(self, eps=1):
    self.__eps = eps

  @property
  def name(self):
    return 'EpsilonGreedy'

  def _learner_init(self):
    pass

  def _learner_choice(self, context):
    """return an arm to pull"""
    if self._t <= self._arm_num:
      return (self._t-1) % self._arm_num

    rand = np.random.random_sample()
    if rand <= self.__eps/self._t:
      return np.random.randint(self._arm_num)
    return np.argmax(np.array([arm.em_mean for arm in self._em_arms]))

  def _learner_update(self, context, action, feedback):
    pass