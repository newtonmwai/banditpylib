import google.protobuf.text_format as text_format

from banditpylib.data_pb2 import Context, Feedback
from .explore_then_commit_batch import ExploreThenCommitBatch


class TestEXP3:
    """Test EXP3 policy"""

    def test_simple_run(self):
        arm_num = 5
        horizon = 10
        learner = ExploreThenCommitBatch(arm_num=arm_num, T_prime=6)
        learner.reset()

        for _ in range(1, horizon + 1):
            actions = learner.actions(Context())
            assert len(actions.arm_pulls) == 1
            arm_pull = actions.arm_pulls[0]
            arm_id = arm_pull.arm.id
            assert arm_pull.times == 1
            learner.update(
                text_format.Parse(
                    """
        arm_feedbacks <
          arm <
            id: {arm_id}
          >
          rewards: 0
        >
        """.format(
                        arm_id=arm_id
                    ),
                    Feedback(),
                )
            )

  @property
  def best_arm(self) -> int:
    return self.__best_arm