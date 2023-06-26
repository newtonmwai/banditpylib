from typing import Optional

import numpy as np

from banditpylib import argmax_or_min_tuple
from banditpylib.arms import PseudoArm
from banditpylib.data_pb2 import Context, Actions, Feedback
from .utils import MABLearner


class ExploreThenCommitBatch(MABLearner):
    r"""Explore-Then-Commit policy

    During the first :math:`T' \leq T` time steps (exploration period), play each arms T/K times. Then for the remaining time steps, play the arm
    with the maximum empirical mean reward within exploration period consistently.

    :param int arm_num: number of arms
    :param int T_prime: time steps to explore
    :param Optional[str] name: alias name
    """

    def __init__(self, arm_num: int, T_prime: int, name: Optional[str] = None):
        super().__init__(arm_num=arm_num, name=name)
        if T_prime < arm_num:
            raise ValueError("T' is expected at least %d. got %d." % (arm_num, T_prime))
        self.__T_prime = T_prime
        self.__best_arm: int = -1
        self.batch_prop = T_prime // arm_num
        self.batch_modulo = T_prime % arm_num

    def _name(self) -> str:
        return "explore_then_commit"

    def reset(self):
        self.__pseudo_arms = [PseudoArm() for arm_id in range(self.arm_num)]
        # Current time step
        self.__time = 1

    def actions(self, context: Context) -> Actions:
        del context

        actions = Actions()
        arm_pull = actions.arm_pulls.add()
        if self.__time <= (self.__T_prime - self.batch_modulo):
            # arm_pull.arm.id = (self.__time - 1) % self.arm_num
            arm_pull.arm.id = (self.__time - 1) // self.batch_prop

        else:
            arm_pull.arm.id = self.__best_arm

        # print((self.__time - 1), arm_pull.arm.id)
        arm_pull.times = 1
        return actions

    def update(self, feedback: Feedback):
        arm_feedback = feedback.arm_feedbacks[0]
        self.__pseudo_arms[arm_feedback.arm.id].update(np.array(arm_feedback.rewards))
        self.__time += 1
        if self.__best_arm < 0 and self.__time > (self.__T_prime - self.batch_modulo):
            self.__best_arm = argmax_or_min_tuple(
                [
                    (self.__pseudo_arms[arm_id].em_mean, arm_id)
                    for arm_id in range(self.arm_num)
                ]
            )
