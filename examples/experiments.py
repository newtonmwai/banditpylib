import sys

sys.path.append("../")
import argparse

import numpy as np
import pandas as pd
import tempfile
import json

# import matplotlib.pyplot as plt
# import seaborn as sns
# sns.set(style="darkgrid")

import logging

logging.getLogger().setLevel(logging.INFO)

from banditpylib import trials_to_dataframe
from banditpylib.arms import GaussianArm
from banditpylib.bandits import MultiArmedBandit
from banditpylib.protocols import SinglePlayerProtocol
from banditpylib.learners.mab_fcbai_learner import (
    ExpGap,
    LilUCBHeuristic,
    TrackAndStop,
    BatchRacing,
    BatchTrackAndStop,
)
from banditpylib.utils import (
    argmax_or_min_tuple,
    argmax_or_min,
    argmax_or_min_tuple_second,
)


def main(rho=0.5, gamma=0.5, batch_size=50, _means="m1"):
    confidence = 0.99
    means_dict = {
        "m1": [0.7, 0.4, 0.1],
        "m2": [0.9, 0.7, 0.4, 0.1],
        "m3": [0.9, 0.7, 0.5, 0.4, 0.45, 0.4, 0.3, 0.2],
    }

    means = means_dict[_means]
    max_pulls = 50000
    std = 1

    print("rho=", rho, "gamma=", gamma, "batch_size=", batch_size, "means=", means)

    arms = [GaussianArm(mu=mean, std=std) for mean in means]
    bandit = MultiArmedBandit(arms=arms)
    learners = [
        # ExpGap(arm_num=len(arms), confidence=confidence, threshold=3,  name='Exponential-Gap Elimination'),
        # BatchRacing(
        #     arm_num=len(arms),
        #     confidence=confidence,
        #     max_pulls=max_pulls,
        #     k=1,
        #     b=batch_size,
        #     r=int(batch_size / 2),
        #     name="BatchRacing",
        # ),
        BatchTrackAndStop(
            arm_num=len(arms),
            confidence=confidence,
            batch_size=batch_size,
            rho=rho,
            gamma=gamma,
            max_pulls=max_pulls,
            name="Batched Track and stop",
        ),
        LilUCBHeuristic(
            arm_num=len(arms),
            confidence=confidence,
            max_pulls=max_pulls,
            name="Heuristic lilUCB",
        ),
        TrackAndStop(
            arm_num=len(arms),
            confidence=confidence,
            tracking_rule="D",
            max_pulls=max_pulls,
            name="Track and stop D-Tracking",
        ),
        TrackAndStop(
            arm_num=len(arms),
            confidence=confidence,
            tracking_rule="C",
            max_pulls=max_pulls,
            name="Track and stop C-Tracking",
        ),
    ]

    # For each setup, we run 20 trials
    trials = 100
    temp_file = tempfile.NamedTemporaryFile()

    game = SinglePlayerProtocol(bandit=bandit, learners=learners)
    # Start playing the game
    # Add `debug=True` for debugging purpose
    game.play(trials=trials, output_filename=temp_file.name)

    trials_df = trials_to_dataframe(temp_file.name)
    trials_df.to_csv(
        "csv_files/trial_df_means_"
        + str(_means)
        + "_rho_"
        + str(rho)
        + "_gamma_"
        + str(gamma)
        + "_batch_size_"
        + str(batch_size)
        + "_.csv",
        index=False,
    )  # `index=False` ensures that the index is not saved in the CSV.


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run rho experiments")
    parser.add_argument("-r", "--rho", type=float, default=0.5, dest="rho")
    parser.add_argument("-g", "--gamma", type=float, default=0.5, dest="gamma")
    parser.add_argument("-b", "--batch_size", type=int, default=50, dest="batch_size")
    parser.add_argument("-m", "--means", type=str, default="m1", dest="means")
    args = parser.parse_args()
    main(
        rho=float(args.rho),
        gamma=float(args.gamma),
        batch_size=int(args.batch_size),
        _means=str(args.means),
    )
