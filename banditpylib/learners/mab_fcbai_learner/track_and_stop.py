from typing import Optional

import numpy as np
from scipy.optimize import minimize_scalar, minimize, root_scalar, bisect


from banditpylib import (
    argmax_or_min_tuple,
    argmax_or_min_tuple_second,
    subtract_tuple_lists,
    add_tuple_lists,
    kl_divergence,
    remove_array_item_,
    remove_tuple_element_,
)
from banditpylib.arms import PseudoArm
from banditpylib.data_pb2 import Context, Actions, Feedback
from .utils import MABFixedConfidenceBAILearner

# Implement the track and stop algorithm from Garivier and Kaufmann 2016
class TrackAndStop(MABFixedConfidenceBAILearner):
    """_summary_

    Args:
        MABFixedConfidenceBAILearner (_type_): _description_
    """

    def __init__(
        self,
        arm_num: int,
        confidence: float,
        max_pulls: int = 1000,
        tracking_rule: str = "D",
        name: Optional[str] = None,
    ):
        super().__init__(arm_num=arm_num, confidence=confidence, name=name)
        self.tracking_rule = tracking_rule
        self.__max_pulls = max_pulls
        self.__eps = 1e-32
        self.__eps_ = 1e32

    def _name(self) -> str:
        return "track_and_stop"

    def reset(self):
        self.__pseudo_arms = [PseudoArm() for arm_id in range(self.arm_num)]
        # Total number of pulls used
        self.__total_pulls = 0
        self.__stage = "initialization"
        self.__stop = False
        self.mu_hat = np.zeros(self.arm_num)
        self.__best_arm_id = np.random.choice([i for i in range(self.arm_num)])
        self.__second_best_arm_id = np.random.choice(
            [i for i in range(self.arm_num) if i != self.__best_arm_id]
        )
        self.Na_t = [(0, arm_id) for arm_id in range(self.arm_num)]
        self.__Na_t = list(np.zeros(self.arm_num))

    # L∞ projection of w_star onto Σ_K with ε = (K^2 + t)^(-1/2)
    def l_inf_projection(self, w_star):
        epsilon = (self.arm_num**2 + self.__total_pulls) ** (
            -1 / 2
        )  # Compute epsilon
        projection = np.clip(w_star, 0, epsilon)  # Apply element-wise clipping
        projection = projection / np.sum(projection)
        return projection

    # Define the function ga(x)
    def ga(self, x, mu, a):
        return kl_divergence(
            mu[self.__best_arm_id], (mu[self.__best_arm_id] + (x * mu[a])) / (1 + x)
        ) + x * kl_divergence(mu[a], (mu[self.__best_arm_id] + (x * mu[a])) / (1 + x))

    # Approximate numerically the inverse function of ga(y)
    def xa(self, y, mu, a):
        def objective(x):
            return np.abs(self.ga(x, mu, a) - y)

        bounds = [(0, None)]  # kl_divergence(mu[self.__best_arm_id], mu[a])
        result = 1
        if a != self.__best_arm_id:
            result = minimize(
                objective,
                x0=0.5 * kl_divergence(mu[self.__best_arm_id], mu[a]),
                bounds=bounds,
                method="SLSQP",  # or method="L-BFGS-B", method='trust-constr' or method="SLSQP"
            ).x[0]
        return result

    # Compute F_mu(y)
    def F_mu(self, y, mu):
        return sum(
            [
                (
                    kl_divergence(
                        mu[self.__best_arm_id],
                        (
                            (mu[self.__best_arm_id] + self.xa(y, mu, a) * mu[a])
                            / (1 + self.xa(y, mu, a))
                        ),
                    )
                    / kl_divergence(
                        mu[a],
                        (
                            (mu[self.__best_arm_id] + self.xa(y, mu, a) * mu[a])
                            / (1 + self.xa(y, mu, a))
                        ),
                    )
                )
                for a in [
                    idx for idx in range(self.arm_num) if idx != self.__best_arm_id
                ]
            ]
        )

    def solve_wstar(self, mu):
        def objective(y):
            return np.abs(self.F_mu(y, mu) - 1)

        bounds = [
            (0, kl_divergence(mu[self.__best_arm_id], mu[self.__second_best_arm_id]))
        ]  # None, kl_divergence(mu[self.__best_arm_id], mu[self.__second_best_arm_id]))

        result = minimize(
            objective,
            x0=kl_divergence(mu[self.__best_arm_id], mu[self.__second_best_arm_id]) / 2,
            bounds=bounds,
            method="SLSQP",  # or method="L-BFGS-B", method='trust-constr' or method="SLSQP"
        )
        y_star = result.x[0]

        # Compute w_star_a for every a in A
        w_star = np.array(
            [self.xa(y_star, mu, a) for a in range(self.arm_num)]
        ).squeeze()

        w_star = np.array(w_star) / sum(w_star)
        # print("w_star: ", w_star)

        return w_star

    def beta(self):
        """Computes Threshold β(t, δ) = log((log(t) + 1)/δ)"""
        return np.log((np.log(self.__total_pulls) + 1.0) / (1.0 - self.confidence))

    def mu_hat_a_b(self, a, b):
        """Compute the weighted average of the empirical means of arms a and b"""
        return (
            ((self.mu_hat[a] * self.__Na_t[a]) + (self.mu_hat[b] * self.__Na_t[b]))
        ) / (self.__Na_t[a] + self.__Na_t[b])

    def Z_a_b(self):
        Z_a_b = np.array(
            [
                (
                    self.__Na_t[self.__best_arm_id]
                    * kl_divergence(
                        self.mu_hat[self.__best_arm_id],
                        self.mu_hat_a_b(self.__best_arm_id, a),
                    )
                )
                + (
                    self.__Na_t[a]
                    * kl_divergence(
                        self.mu_hat[a],
                        self.mu_hat_a_b(self.__best_arm_id, a),
                    )
                )
                for a in [
                    idx for idx in range(self.arm_num) if idx != self.__best_arm_id
                ]
            ]
        )
        return Z_a_b

    def stop(self):
        """Stop the algorithm if the stopping condition is satisfied"""
        __Z_a_b = self.Z_a_b()
        __beta = self.beta()
        # print("Z_a_b: ", __Z_a_b)
        # print("Beta: ", __beta)
        return np.min(np.array(__Z_a_b)) > __beta

    def actions(self, context: Context) -> Actions:
        if self.tracking_rule == "D":

            if self.__stage == "initialization":
                # print("Initialization")
                actions = Actions()
                for arm_id in range(self.arm_num):
                    # print("Pulling arm ", arm_id)
                    arm_pull = actions.arm_pulls.add()
                    arm_pull.arm.id = arm_id
                    arm_pull.times = 1
                return actions

            self.__stage == "main"
            actions = Actions()

            # Repeat until stopping condition is met
            if self.__stop or self.__total_pulls >= self.__max_pulls:
                return actions

            # Forced exploration (if Ut ≠ ∅)
            if len(self.__U_t) > 0:
                arm_pull = actions.arm_pulls.add()
                arm_pull.arm.id = argmax_or_min_tuple(self.__U_t, True)
                arm_pull.times = 1
                return actions

            # Compute w_star
            w_star = self.solve_wstar(self.mu_hat)
            # print("w_star: ", w_star)

            # Pull an arm according to the tracking rule
            t_wstar = [(self.__total_pulls * w_star[a], a) for a in range(self.arm_num)]

            arm_id = argmax_or_min_tuple(subtract_tuple_lists(t_wstar, self.Na_t))
            arm_pull = actions.arm_pulls.add()
            arm_pull.arm.id = arm_id
            arm_pull.times = 1

            return actions

        elif self.tracking_rule == "C":
            actions = Actions()

            if self.__stop or self.__total_pulls >= self.__max_pulls:
                return actions

            # Compute w_star
            w_star = self.solve_wstar(self.mu_hat)

            if self.__total_pulls == 0:
                print("w_star: ", w_star)

            inf_wstar = self.l_inf_projection(w_star)
            inf_wstar = [(inf_wstar[a], a) for a in range(self.arm_num)]

            if self.__total_pulls == 0:
                self.__inf_wstar = inf_wstar
            else:
                self.__inf_wstar = add_tuple_lists(self.__inf_wstar, inf_wstar)

            arm_id = argmax_or_min_tuple(
                subtract_tuple_lists(self.__inf_wstar, self.Na_t)
            )

            arm_pull = actions.arm_pulls.add()
            arm_pull.arm.id = arm_id
            arm_pull.times = 1

            return actions

    def update(self, feedback: Feedback):
        for arm_feedback in feedback.arm_feedbacks:
            self.__pseudo_arms[arm_feedback.arm.id].update(
                np.array(arm_feedback.rewards)
            )
            # Update tracking statistics here
            self.__total_pulls += len(arm_feedback.rewards)

        self.__best_arm_id = self.best_arm
        self.__second_best_arm_id = self.second_best_arm

        # Compute mu_hat
        self.mu_hat = [
            self.__pseudo_arms[arm_id].em_mean
            if self.__pseudo_arms[arm_id].total_pulls > 0
            else self.mu_hat[arm_id]
            for arm_id in range(self.arm_num)
        ]
        self.Na_t = [
            (pseudo_arm.total_pulls, arm_id)
            for (arm_id, pseudo_arm) in enumerate(self.__pseudo_arms)
        ]

        self.__Na_t = [
            self.__pseudo_arms[arm_id].total_pulls for arm_id in range(self.arm_num)
        ]

        self.__U_t = []
        for arm_id in range(self.arm_num):
            if self.__pseudo_arms[arm_id].total_pulls < (
                np.sqrt(self.__total_pulls) - self.arm_num / 2
            ):
                self.__U_t = [
                    (pseudo_arm.total_pulls, __arm_id)
                    for (__arm_id, pseudo_arm) in enumerate(self.__pseudo_arms)
                ]
        self.__stop = self.stop()

        if self.__stage == "initialization":
            self.__stage = "main"

    @property
    def best_arm(self) -> int:
        return argmax_or_min_tuple(
            [(self.mu_hat[arm_id], arm_id) for arm_id in range(self.arm_num)]
        )

    @property
    def second_best_arm(self) -> int:
        best_arm_id = self.best_arm
        return argmax_or_min_tuple(
            [
                (self.mu_hat[arm_id], arm_id)
                for arm_id in range(self.arm_num)
                if arm_id != best_arm_id
            ]
        )
