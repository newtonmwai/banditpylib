from typing import List, Tuple
import numpy as np
import pandas as pd
import google.protobuf.json_format as json_format
from google.protobuf.internal.decoder import _DecodeVarint32
from banditpylib.data_pb2 import Trial


def argmax_or_min(values: List[float], find_min: bool = False) -> int:
    """Find index with the largest or smallest value

    Args:
        values: a list of values
        find_min: whether to select smallest value

    Returns:
        index with the largest or smallest value. When there is a tie, randomly
        output one of the indexes.
    """
    extremum = min(values) if find_min else max(values)
    indexes = [index for index, value in enumerate(values) if value == extremum]
    return np.random.choice(indexes)


def argmax_or_min_tuple(values: List[Tuple[float, int]], find_min: bool = False) -> int:
    """Find the element of the tuple with the largest or smallest value

    Args:
        values: a list of tuples
        find_min: whether to select smallest value

    Returns:
        the element of the tuple with the largest or smallest value.
        When there is a tie, randomly output one of them.
    """
    extremum = (
        min(values, key=lambda x: x[0])[0]
        if find_min
        else max(values, key=lambda x: x[0])[0]
    )
    indexes = [index for value, index in values if value == extremum]
    return np.random.choice(indexes)


def argmax_or_min_tuple_second(
    values: List[Tuple[float, int]], find_min: bool = False
) -> int:
    """Find the element of the tuple with the second largest or smallest value

    Args:
        values: a list of tuples
        find_min: whether to select smallest value

    Returns:
        the element of the tuple with the second largest or smallest value.
        When there is a tie, randomly output one of them.
    """
    extremum = (
        min(values, key=lambda x: x[0])[0]
        if find_min
        else max(values, key=lambda x: x[0])[0]
    )

    sorted_values = sorted(values, key=lambda x: x[0])
    second_extremum = sorted_values[1][0] if find_min else sorted_values[-2][0]

    indexes = [index for value, index in values if value == second_extremum]
    return np.random.choice(indexes)


def parse_trials_from_bytes(data: bytes) -> List[Trial]:
    """Parse trials from bytes

    Args:
        data: bytes data

    Returns:
        trial protobuf messages
    """
    trials = []
    next_pos, pos = 0, 0
    while pos < len(data):
        trial = Trial()
        next_pos, pos = _DecodeVarint32(data, pos)
        trial.ParseFromString(data[pos : pos + next_pos])
        pos += next_pos
        trials.append(trial)
    return trials


def trials_to_dataframe(filename: str) -> pd.DataFrame:
    """Read bytes file storing trials and transform to pandas DataFrame

    Args:
        filename: file name

    Returns:
        pandas dataframe
    """
    data = []
    with open(filename, "rb") as f:
        trials = parse_trials_from_bytes(f.read())
        for trial in trials:
            for result in trial.results:
                tmp_dict = json_format.MessageToDict(
                    result,
                    including_default_value_fields=True,
                    preserving_proto_field_name=True,
                )
                tmp_dict["bandit"] = trial.bandit
                tmp_dict["learner"] = trial.learner
                data.append(tmp_dict)
    data_df = pd.DataFrame.from_dict(data)
    return data_df


def subtract_tuple_lists(
    list1: List[Tuple[float, int]], list2: List[Tuple[float, int]]
) -> List[Tuple[float, int]]:
    """
    Subtract two lists of tuples
        Args:
            list1: a list of tuples
            list2: a list of tuples
        Returns:
            a list of tuples
    """
    result = [
        (value1 - value2, key) for (value1, key), (value2, _) in zip(list1, list2)
    ]
    return result


def add_tuple_lists(
    list1: List[Tuple[float, int]], list2: List[Tuple[float, int]]
) -> List[Tuple[float, int]]:
    """
    Add two lists of tuples
        Args:
            list1: a list of tuples
            list2: a list of tuples
        Returns:
            a list of tuples
    """
    result = [
        (value1 + value2, key) for (value1, key), (value2, _) in zip(list1, list2)
    ]
    return result


# Define the KL-divergence function
def kl_divergence(mean1, mean2, var1=1, var2=1):
    """Compute the KL divergence between two Gaussian distributions.

    Args:
        mean1: mean of the first Gaussian distribution.
        var1: variance of the first Gaussian distribution.
        mean2: mean of the second Gaussian distribution.
        var2: variance of the second Gaussian distribution.

    Returns:
        The KL divergence between the two Gaussian distributions.
    """
    kl = np.log(var2 / var1) + (var1 + (mean1 - mean2) ** 2) / (2 * var2) - 0.5
    return kl


def remove_array_item_(array, index):
    if 0 <= index < len(array):
        array.pop(index)


def remove_tuple_element_(lst, key):
    return [item for item in lst if item[1] != key]


def round_robin(A, T, b, r):
    """
    A RoundRobin procedure to distribute arm pulls uniformly.

    Args:
    - A: a set of arms
    - T: arm pull counts corresponding to the arms in A
    - b: batch size
    - r: repeated pull limit

    Returns:
    - a: pull count vector for the arms
    """

    a = [(arm_id, 0) for arm_id in A]

    for _ in range(min(b, len(A) * r)):
        j = np.argmin([T[i] + a[i][1] if a[i][1] <= r else float("inf") for i in A])
        lst = list(a[j])
        lst[1] += 1
        a[j] = tuple(lst)

    return a
