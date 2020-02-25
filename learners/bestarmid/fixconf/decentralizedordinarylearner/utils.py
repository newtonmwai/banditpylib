from abc import abstractmethod

from absl import logging

from bandits.arms import EmArm
from learners.bestarmid.fixconf import FixConfBAILearner

__all__ = ['DecentralizedOrdinaryLearner']


class DecentralizedOrdinaryLearner(FixConfBAILearner):
  """base class for learners in the classic bandit model"""

  # pylint: disable=I0023, W0235
  def __init__(self, pars):
    super().__init__(pars)

  @abstractmethod
  def _learner_init(self):
    pass

  @abstractmethod
  # pylint: disable=arguments-differ
  def learner_choice(self, messages):
    pass

  @abstractmethod
  def broadcast_message(self, action, feedback):
    pass

  @abstractmethod
  def learner_run(self):
    pass

  @abstractmethod
  def best_arm(self):
    pass

  def _model_init(self):
    """local initialization"""
    if self._bandit.type != 'ordinarybandit':
      logging.fatal(("(%s) I don't understand",
                     " the bandit environment!") % self.name)
    self._arm_num = self._bandit.arm_num
    # record empirical information for every arm
    self._em_arms = [EmArm() for ind in range(self._arm_num)]

  def _model_update(self, action, feedback):
    if isinstance(action, list):
      for (i, tup) in enumerate(action):
        self._em_arms[tup[0]].update(feedback[0][i], tup[1])
    else:
      self._em_arms[action].update(feedback[0])

  def update(self, action, feedback):
    self._model_update(action, feedback)