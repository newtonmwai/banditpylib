"""
@generated by mypy-protobuf.  Do not edit manually!
isort:skip_file
"""
import builtins
import google.protobuf.descriptor
import google.protobuf.internal.containers
import google.protobuf.message
import typing
import typing_extensions

DESCRIPTOR: google.protobuf.descriptor.FileDescriptor = ...

class Arm(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor = ...
    ID_FIELD_NUMBER: builtins.int
    IDS_FIELD_NUMBER: builtins.int
    id: builtins.int = ...
    ids: google.protobuf.internal.containers.RepeatedScalarFieldContainer[builtins.int] = ...

    def __init__(self,
        *,
        id : builtins.int = ...,
        ids : typing.Optional[typing.Iterable[builtins.int]] = ...,
        ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal[u"id",b"id",u"ids",b"ids"]) -> None: ...
global___Arm = Arm

class ArmPullsPair(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor = ...
    ARM_FIELD_NUMBER: builtins.int
    PULLS_FIELD_NUMBER: builtins.int
    pulls: builtins.int = ...

    @property
    def arm(self) -> global___Arm: ...

    def __init__(self,
        *,
        arm : typing.Optional[global___Arm] = ...,
        pulls : builtins.int = ...,
        ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal[u"arm",b"arm"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal[u"arm",b"arm",u"pulls",b"pulls"]) -> None: ...
global___ArmPullsPair = ArmPullsPair

class Actions(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor = ...
    ARM_PULLS_PAIRS_FIELD_NUMBER: builtins.int

    @property
    def arm_pulls_pairs(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[global___ArmPullsPair]: ...

    def __init__(self,
        *,
        arm_pulls_pairs : typing.Optional[typing.Iterable[global___ArmPullsPair]] = ...,
        ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal[u"arm_pulls_pairs",b"arm_pulls_pairs"]) -> None: ...
global___Actions = Actions

class ArmRewardsPair(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor = ...
    ARM_FIELD_NUMBER: builtins.int
    REWARDS_FIELD_NUMBER: builtins.int
    CUSTOMER_FEEDBACKS_FIELD_NUMBER: builtins.int
    rewards: google.protobuf.internal.containers.RepeatedScalarFieldContainer[builtins.float] = ...
    customer_feedbacks: google.protobuf.internal.containers.RepeatedScalarFieldContainer[builtins.int] = ...

    @property
    def arm(self) -> global___Arm: ...

    def __init__(self,
        *,
        arm : typing.Optional[global___Arm] = ...,
        rewards : typing.Optional[typing.Iterable[builtins.float]] = ...,
        customer_feedbacks : typing.Optional[typing.Iterable[builtins.int]] = ...,
        ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal[u"arm",b"arm"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal[u"arm",b"arm",u"customer_feedbacks",b"customer_feedbacks",u"rewards",b"rewards"]) -> None: ...
global___ArmRewardsPair = ArmRewardsPair

class Feedback(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor = ...
    ARM_REWARDS_PAIRS_FIELD_NUMBER: builtins.int

    @property
    def arm_rewards_pairs(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[global___ArmRewardsPair]: ...

    def __init__(self,
        *,
        arm_rewards_pairs : typing.Optional[typing.Iterable[global___ArmRewardsPair]] = ...,
        ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal[u"arm_rewards_pairs",b"arm_rewards_pairs"]) -> None: ...
global___Feedback = Feedback

class DataItem(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor = ...
    ROUNDS_FIELD_NUMBER: builtins.int
    TOTAL_ACTIONS_FIELD_NUMBER: builtins.int
    REGRET_FIELD_NUMBER: builtins.int
    OTHER_FIELD_NUMBER: builtins.int
    rounds: builtins.int = ...
    total_actions: builtins.int = ...
    regret: builtins.float = ...
    other: builtins.float = ...

    def __init__(self,
        *,
        rounds : builtins.int = ...,
        total_actions : builtins.int = ...,
        regret : builtins.float = ...,
        other : builtins.float = ...,
        ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal[u"other",b"other",u"regret",b"regret",u"rounds",b"rounds",u"total_actions",b"total_actions"]) -> None: ...
global___DataItem = DataItem

class Trial(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor = ...
    BANDIT_FIELD_NUMBER: builtins.int
    LEARNER_FIELD_NUMBER: builtins.int
    DATA_ITEMS_FIELD_NUMBER: builtins.int
    bandit: typing.Text = ...
    learner: typing.Text = ...

    @property
    def data_items(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[global___DataItem]: ...

    def __init__(self,
        *,
        bandit : typing.Text = ...,
        learner : typing.Text = ...,
        data_items : typing.Optional[typing.Iterable[global___DataItem]] = ...,
        ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal[u"bandit",b"bandit",u"data_items",b"data_items",u"learner",b"learner"]) -> None: ...
global___Trial = Trial
