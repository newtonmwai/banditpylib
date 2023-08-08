from typing import Optional

import numpy as np

from banditpylib import argmax_or_min_tuple, round_robin
from banditpylib.arms import PseudoArm
from banditpylib.data_pb2 import Context, Actions, Feedback
from .utils import MABFixedConfidenceBAILearner


class BatchRacing(MABFixedConfidenceBAILearner):
    """inproceedings{jun2016top,
      title={Top arm identification in multi-armed bandits with batch arm pulls},
      author={Jun, Kwang-Sung and Jamieson, Kevin and Nowak, Robert and Zhu, Xiaojin},
      booktitle={Artificial Intelligence and Statistics},
      pages={139--148},
      year={2016},
      organization={PMLR}
    }
        :param int arm_num: number of arms
        :param float confidence: confidence level. It should be within (0, 1). The
          algorithm should output the best arm with probability at least this value.
        :param Optional[str] name: alias name
        :param int max_pulls: maximum number of pulls
        :param int  b: batch size
        :param int r: repeated pull limit
        :param int k: number of top arms
    """

    def __init__(
        self,
        arm_num: int,  # n
        confidence: float,
        max_pulls: int,
        k: int,
        b: int,
        r: int,
        name: Optional[str] = None,
    ):
        super().__init__(arm_num=arm_num, confidence=confidence, name=name)
        self.__max_pulls = max_pulls
        self.__k = k
        self.__b = b
        self.__r = r

    def _name(self) -> str:
        return "batch_racing"

    def reset(self):
        self.__pseudo_arms = [PseudoArm() for arm_id in range(self.arm_num)]
        self.__total_pulls = 0
        self.S = set(range(self.arm_num))
        self.R = set()
        self.A = set()
        self.omega = np.sqrt((1 - self.confidence) / (6 * self.arm_num))
        self.__lcb = np.zeros(self.arm_num)
        self.__ucb = np.zeros(self.arm_num)

    def __update_LCB(self, arm_id):
        # Update the Lower Confidence Bound
        self.__lcb[arm_id] = self.__pseudo_arms[arm_id].em_mean - self.D(
            self.__pseudo_arms[arm_id].total_pulls, self.omega
        )

    def __update_UCB(self, arm_id):
        # Update the Upper Confidence Bound
        self.__ucb[arm_id] = self.__pseudo_arms[arm_id].em_mean + self.D(
            self.__pseudo_arms[arm_id].total_pulls, self.omega
        )

    def D(self, tau, omega):
        return np.sqrt(4 * np.log(np.log2(2 * tau) / omega) / tau)

    def actions(self, context: Context) -> Actions:
        if self.__total_pulls >= self.__max_pulls:
            actions = Actions()
            return actions

        if len(self.A) < self.__k:
            actions = Actions()
            __a = round_robin(
                self.S,
                [self.__pseudo_arms[i].total_pulls for i in self.S],
                self.__b,
                self.__r,
            )
            # print("__arms", __a)
            # print("self.__ucb: ", self.__ucb)
            # print("self.__lcb: ", self.__lcb)
            for __arm in __a:
                arm_pull = actions.arm_pulls.add()
                arm_pull.arm.id = __arm[0]
                arm_pull.times = __arm[1]
            return actions
        else:
            actions = Actions()
            # for arm_id in self.A:
            #     print("arm_id: ", arm_id)
            #     arm_pull = actions.arm_pulls.add()
            #     arm_pull.arm.id = arm_id
            #     arm_pull.times = 1
            # arm_pull = actions.arm_pulls.add()
            # arm_pull.arm.id = [arm_id for arm_id in self.A][0]
            # arm_pull.times = 1
            return actions

    def update(self, feedback: Feedback):
        for arm_feedback in feedback.arm_feedbacks:
            self.__pseudo_arms[arm_feedback.arm.id].update(
                np.array(arm_feedback.rewards)
            )
            self.__update_LCB(arm_feedback.arm.id)
            self.__update_UCB(arm_feedback.arm.id)
            self.__total_pulls += len(arm_feedback.rewards)

        k_t = self.__k - len(self.A)
        A_next = self.A.copy()
        for i in self.S:
            if (
                self.__lcb[i]
                > max([self.__ucb[j] for j in self.S if j not in A_next][: k_t + 1])
                or len(self.S) <= self.__k
            ):
                A_next.add(i)
                print("A_next: ", A_next)

        R_next = self.R.copy()
        for i in self.S:
            if self.__ucb[i] < max(
                [self.__lcb[j] for j in self.S if j not in R_next][:k_t]
            ):
                R_next.add(i)
                print("R_next: ", R_next)

        self.S = self.S - A_next - R_next
        self.A = A_next
        self.R = R_next

        # if self.__stage == "initialization":
        #     self.__stage = "main"

    @property
    def best_arm(self) -> int:
        return argmax_or_min_tuple(
            [
                (pseudo_arm.total_pulls, arm_id)
                for (arm_id, pseudo_arm) in enumerate(self.__pseudo_arms)
            ]
        )
