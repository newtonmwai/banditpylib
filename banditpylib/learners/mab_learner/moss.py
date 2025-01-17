from typing import Optional

import numpy as np

from banditpylib.arms import PseudoArm
from banditpylib.data_pb2 import Context, Actions, Feedback
from .utils import MABLearner


class MOSS(MABLearner):
    r"""MOSS policy :cite:`audibert2009minimax`

    At time :math:`t`, play arm

    .. math::
      \mathrm{argmax}_{i \in \{0, \dots, N-1\}} \left\{ \bar{\mu}_i(t) +
      \sqrt{\frac{\mathrm{max}(\ln( \frac{T}{N T_i(t)} ), 0 ) }{T_i(t)} } \right\}

    :param int arm_num: number of arms
    :param int horizon: total number of time steps
    :param Optional[str] name: alias name

    .. note::
      MOSS uses time horizon in its confidence interval. Reward has to be bounded
      in [0, 1].
    """

    def __init__(self, arm_num: int, horizon: int, name: Optional[str] = None):
        super().__init__(arm_num=arm_num, name=name)
        if horizon < arm_num:
            raise Exception(
                "Horizon is expected at least %d. Got %d." % (arm_num, horizon)
            )
        self.__horizon = horizon

    def _name(self) -> str:
        return "moss"

    def reset(self):
        self.__pseudo_arms = [PseudoArm() for arm_id in range(self.arm_num)]
        # Current time step
        self.__time = 1

    def __MOSS(self) -> np.ndarray:
        """
        Returns:
          optimistic estimate of arms' real means
        """
        moss = np.array(
            [
                arm.em_mean
                + np.sqrt(
                    np.maximum(
                        0, np.log(self.__horizon / (self.arm_num * arm.total_pulls))
                    )
                    / arm.total_pulls
                )
                for arm in self.__pseudo_arms
            ]
        )
        return moss

    def actions(self, context: Context) -> Actions:
        del context

        actions = Actions()
        arm_pull = actions.arm_pulls.add()

        if self.__time <= self.arm_num:
            arm_pull.arm.id = self.__time - 1
        else:
            arm_pull.arm.id = int(np.argmax(self.__MOSS()))

        arm_pull.times = 1
        return actions

    def update(self, feedback: Feedback):
        arm_feedback = feedback.arm_feedbacks[0]
        self.__pseudo_arms[arm_feedback.arm.id].update(np.array(arm_feedback.rewards))
        self.__time += 1
