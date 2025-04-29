from dataclasses import dataclass


@dataclass
class BaseControlConfig:
    # exp
    seed: int = 0
    not_render: bool = False
    # env
    env_name: str = (
        "ant"  # "humanoidstandup", "ant", "halfcheetah", "hopper", "walker2d", "car2d"
    )
    
class ControlPublisher():
    def __init__(config : BaseControlConfig):
        NotImplementedError

