from typing import Optional

import numpy as np
import pickle
from datetime import datetime
from scipy.optimize import minimize

from banditpylib import (
    argmax_or_min_tuple,
    subtract_tuple_lists,
    kl_divergence,
    k_largest_indices,
)

from banditpylib.arms import PseudoArm
from banditpylib.data_pb2 import Context, Actions, Feedback
from .utils import MABFixedConfidenceBAILearner

# Implement the track and stop algorithm from Garivier and Kaufmann 2016
class BatchTrackAndStop(MABFixedConfidenceBAILearner):
    """_summary_

    Args:
        MABFixedConfidenceBAILearner (_type_): _description_
    """

    def __init__(
        self,
        arm_num: int,
        confidence: float,
        batch_size: int,
        max_pulls: int = 1000,
        num_switches=1,
        gamma: float = 0.5,
        name: Optional[str] = None,
    ):
        super().__init__(arm_num=arm_num, confidence=confidence, name=name)
        self.__max_pulls = max_pulls
        # self.__rho = rho
        self.batch_size = batch_size
        self.__eps = 1e-16
        self.__sparsity_eps = 1e-5
        self.num_switches = num_switches
        self.desired_sparsity = 1 - ((1.0 + num_switches) / self.arm_num)
        self.max_iterations = 10
        # self.rho = rho
        # self.__rho = rho
        self.gamma = gamma

        self.sparsity = []

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
        self.sparsity = []

    # L∞ projection of w_star onto Σ_K with ε = (K^2 + t)^(-1/2)
    def l_inf_projection(self, w_star):
        epsilon = (self.arm_num**2 + self.__total_pulls) ** (
            -1 / 2
        )  # Compute epsilon
        projection = np.clip(w_star, 0, epsilon)  # Apply element-wise clipping
        projection = projection / np.sum(projection)
        return projection

    def linf_projection(self, w_star):
        # Clip the values of w to the interval [epsilon, 1]
        epsilon = (self.arm_num**2 + self.__total_pulls) ** (
            -1 / 2
        )  # Compute epsilon

        w_proj = np.clip(w_star, epsilon, 1)

        # Calculate discrepancy
        discrepancy = 1 - np.sum(w_proj)

        # While we have a discrepancy to distribute
        while (
            abs(discrepancy) > 1e-8
        ):  # A small threshold to prevent endless loop due to floating point errors
            # Find components that can be adjusted
            adjustable_indices = np.where(
                (w_proj > epsilon) & (w_proj + discrepancy / self.arm_num > epsilon)
            )[0]

            if len(adjustable_indices) == 0:
                # If no components can be adjusted, distribute discrepancy equally and break
                w_proj += discrepancy / self.arm_num
                break

            # Calculate amount to distribute per adjustable component
            per_component_discrepancy = discrepancy / len(adjustable_indices)

            # Distribute among the adjustable components
            w_proj[adjustable_indices] += per_component_discrepancy

            # Recalculate discrepancy
            discrepancy = 1 - np.sum(w_proj)

        return w_proj

    def clip_values(self, w_star, threshold=0.05):
        w_star = np.where(np.array(w_star) < threshold, 0, w_star)
        return w_star / np.sum(w_star)

    def ucb(self, arm_id):
        return np.sqrt(
            2 * np.log(self.__total_pulls) / (self.__Na_t[arm_id] + self.__eps)
        )

    # Define I_alpha
    def I_alpha(self, alpha, mu1, mu2):
        term1 = alpha * kl_divergence(mu1, alpha * mu1 + (1 - alpha) * mu2)
        term2 = (1 - alpha) * kl_divergence(mu2, alpha * mu1 + (1 - alpha) * mu2)
        return term1 + term2

    def solve_wstar(self, mu):
        """
        Solve for the optimal weights given mu values while achieving a desired sparsity level.

        Parameters:
        - mu: list of expected reward values for each arm.
        - desired_sparsity: Target sparsity level for the weights.

        Returns:
        - optimal weights array.
        """

        def objective(params, mu):
            w = params[:-1]
            r = params[-1]
            terms = [
                (w[self.__best_arm_id] + w[i])
                * self.I_alpha(
                    w[self.__best_arm_id] / (w[self.__best_arm_id] + w[i]),
                    mu[self.__best_arm_id] + self.ucb(self.__best_arm_id),
                    mu[i] + self.ucb(i),
                )
                for i in range(len(w))
                if i != self.__best_arm_id
            ]
            return -min(terms) + r

        def constraint1(params):
            w = params[:-1]
            return np.sum(w) - 1

        def constraint2(params, a):
            w = params[:-1]
            r = params[-1]
            return w[a] - self.__gamma / r

        # Callback function
        def adjust_gamma(params):
            w = params[:-1]
            # Measure the sparsity
            sparsity = (
                np.sum(np.array([1.0 for _w in w if _w < self.__sparsity_eps]))
                / self.arm_num
            )
            if sparsity < self.desired_sparsity:
                self.__gamma *= 1.1

        best_weights = None
        best_sparsity = 0

        w0 = np.ones(self.arm_num) / self.arm_num
        r0 = 1
        initial_guess = np.append(w0, r0)

        w_values = np.zeros((self.arm_num, w0.shape[0]))
        fx_values = np.zeros((self.arm_num))

        for a in range(self.arm_num):
            self.__gamma = self.gamma
            constraints = [
                {"type": "eq", "fun": constraint1},
                {"type": "ineq", "fun": constraint2, "args": (a,)},
            ]
            bounds = [(0, 1) for _ in range(self.arm_num)] + [(0, None)]

            result = minimize(
                objective,
                initial_guess,
                args=(mu,),
                constraints=constraints,
                bounds=bounds,
                method="SLSQP",
                callback=adjust_gamma,
            )

            if result.success:
                # print("Success ")
                w_values[a] = result.x[:-1]  # Save weights
                fx_values[a] = result.fun  # Save objective function value
            else:
                w_values[a] = w0
                fx_values[a] = np.inf

        # find index of minimum objective function value
        min_index = np.argmin(fx_values)
        # print("fx_values: ", fx_values)
        # print("w_values: ", w_values)
        best_weights = w_values[min_index]
        best_sparsity = np.mean(best_weights < self.__sparsity_eps)
        # print("Target sparsity: ", self.desired_sparsity)
        # print("Best sparsity: ", best_sparsity)
        # print("gamma: ", self.__gamma)

        # If optimization didn't succeed for any arm, return the initial weights (equal distribution)
        # if best_weights is None:
        #     print("Failure ")
        #     return w0
        # else:
        #     print("Success ")
        return best_weights

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

    def arm_pulls(self, w_star):
        # Multiply w_star with the batch size and round to get integer arm pulls
        pulls = np.round(w_star * self.batch_size).astype(int)

        # Adjust for rounding errors
        while np.sum(pulls) > self.batch_size:
            index_max = np.argmax(pulls)
            pulls[index_max] -= 1

        while np.sum(pulls) < self.batch_size:
            pulls = np.where(pulls == 0, np.inf, pulls)
            index_min = np.argmin(pulls)
            pulls[index_min] += 1
        pulls = np.where(pulls == np.inf, 0, pulls)
        return pulls.astype(int)

    def stop(self):
        """Stop the algorithm if the stopping condition is satisfied"""
        __Z_a_b = self.Z_a_b()
        __beta = self.beta()
        # print("Z_a_b: ", __Z_a_b)
        # print("Beta: ", __beta)
        return np.min(np.array(__Z_a_b)) > __beta

    def actions(self, context: Context) -> Actions:
        # print("Total Arm pulls: ", self.__Na_t)
        actions = Actions()
        w_star = self.solve_wstar(self.mu_hat)
        # __sparsity = (
        #     np.sum(np.array([1.0 for _w in w_star if _w < 1e-3])) / self.arm_num
        # )
        # self.sparsity.append(__sparsity)

        if self.__stop or self.__total_pulls >= self.__max_pulls:
            # print("Final w_star: ", w_star)
            self.save_sparsity_to_file()
            return actions

        __arm_pulls = self.arm_pulls(w_star)

        __sparsity = (
            np.sum(np.array([1.0 for a in __arm_pulls if a == 0])) / self.arm_num
        )
        self.sparsity.append(__sparsity)

        # print("w_star: ", w_star)
        # print("Arm pulls: ", __arm_pulls)
        # print("Total Arm pulls: ", self.__Na_t, "\n")

        for __arm, __pulls in enumerate(__arm_pulls):
            arm_pull = actions.arm_pulls.add()
            arm_pull.arm.id = __arm
            arm_pull.times = __pulls
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

        self.__Na_t = np.array(
            [self.__pseudo_arms[arm_id].total_pulls for arm_id in range(self.arm_num)]
        )

        # self.__U_t = []
        # for arm_id in range(self.arm_num):
        #     if self.__pseudo_arms[arm_id].total_pulls < (
        #         np.sqrt(self.__total_pulls) - self.arm_num / 2
        #     ):
        #         self.__U_t = [
        #             (pseudo_arm.total_pulls, __arm_id)
        #             for (__arm_id, pseudo_arm) in enumerate(self.__pseudo_arms)
        #         ]

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

    def save_sparsity_to_file(self):
        # Get the current timestamp and format it
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        dat = (
            "gamma_"
            + str(self.gamma)
            + "_batch_size_"
            + str(self.batch_size)
            + "num_switches_"
            + str(self.num_switches)
            + "_"
            + current_time
        )
        filename = f"csv_files/data_{dat}.pkl"

        # Write the list to a file using pickle with timestamp in filename
        with open(filename, "wb") as file:
            pickle.dump(self.sparsity, file)
