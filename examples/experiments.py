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


def main(rho=0.5, batch_size=20):
    confidence = 0.95
    means = [0.7, 0.4, 0.1]
    max_pulls = 50000
    std = 1
    batch_size = 20
    rho = 0.5

    # batch_sizes = [20, 50, 100, 500, 1000]
    # rhos= [0.05, 0.25, 0.5, 0.75, 0.95]

    # for batch_size in batch_sizes:
    #     for rho in rhos:
    arms = [GaussianArm(mu=mean, std=std) for mean in means]
    bandit = MultiArmedBandit(arms=arms)
    learners = [
        # ExpGap(arm_num=len(arms), confidence=confidence, threshold=3,  name='Exponential-Gap Elimination'),
        BatchRacing(
            arm_num=len(arms),
            confidence=confidence,
            max_pulls=max_pulls,
            k=1,
            b=batch_size,
            r=int(batch_size / 2),
            name="BatchRacing",
        ),
        BatchTrackAndStop(
            arm_num=len(arms),
            confidence=confidence,
            batch_size=batch_size,
            rho=rho,
            tracking_rule="C",
            max_pulls=max_pulls,
            name="Batch Track and stop C-Tracking",
        ),
        LilUCBHeuristic(
            arm_num=len(arms),
            confidence=confidence,
            max_pulls=max_pulls,
            name="Heuristic lilUCB",
        )
        # TrackAndStop(arm_num=len(arms), confidence=confidence, tracking_rule="C",
        #             max_pulls=max_pulls,  name='Track and stop C-Tracking'),
        # TrackAndStop(arm_num=len(arms), confidence=confidence, tracking_rule="D",
        #             max_pulls=max_pulls,  name='Track and stop D-Tracking')
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
        "csv_files/trial_df_rho_"
        + str(rho)
        + "_batch_size_"
        + str(batch_size)
        + "_.csv",
        index=False,
    )  # `index=False` ensures that the index is not saved in the CSV.


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run rho experiments")
    parser.add_argument("-r", "--rho", type=float, default=0.5, dest="rho")

    parser.add_argument("-b", "--batch_size", type=int, default=20, dest="batch_size")

    args = parser.parse_args()
    main(rho=float(args.rho), batch_size=int(args.batch_size))
