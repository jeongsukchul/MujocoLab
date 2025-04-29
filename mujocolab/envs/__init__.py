from brax import envs as brax_envs

from .pushT import PushT
from .hopper import Hopper
from .humanoidstandup import HumanoidStandup
from .humanoidtrack import HumanoidTrack
from .humanoidrun import HumanoidRun
from .walker2d import Walker2d
from .cartpole import Cartpole
from .car2d import Car2d
from .finger import OneFinger
from .allegro import Allegro
from .allegro import AllegroConfig
from .base_env import BaseEnv
from .unitree_go2_env import UnitreeGo2Env
from .unitree_go2_env import UnitreeGo2EnvConfig
from .unitree_h1_env import UnitreeH1LocoEnv
from .unitree_h1_env import UnitreeH1LocoEnvConfig
from .env_config.base_env_config import BaseEnvCfg
from .env_config.robot_config import RobotEnvCfg

from mujocolab.algorithms.sampling_method.sampling_config import SamplingCfg

def get_env(env_name: str, control_type):
    if env_name == "pushT":
        if control_type == "sampling_method":
            return PushT(), BaseEnvCfg, SamplingCfg
    elif env_name == "hopper":
        if control_type == "sampling_method":
            return Hopper(), BaseEnvCfg, SamplingCfg
    elif env_name == "humanoidstandup":
        if control_type == "sampling_method":
            return HumanoidStandup(), BaseEnvCfg, SamplingCfg
    elif env_name == "humanoidrun":
        if control_type == "sampling_method":
            return HumanoidRun(), BaseEnvCfg, SamplingCfg
    elif env_name == "humanoidtrack":
        if control_type == "sampling_method":
            return HumanoidTrack(), BaseEnvCfg, SamplingCfg
    elif env_name == "walker2d":
        if control_type == "sampling_method":
            return Walker2d(), BaseEnvCfg, SamplingCfg
    elif env_name == "cartpole":
        if control_type == "sampling_method":
            return Cartpole(), BaseEnvCfg, SamplingCfg
    elif env_name == "car2d":
        if control_type == "sampling_method":
            return Car2d(), BaseEnvCfg, SamplingCfg
    elif env_name == "finger":
        if control_type == "sampling_method":
            return OneFinger(), BaseEnvCfg, SamplingCfg
    elif env_name == "allegro":
        if control_type == "sampling_method":
            return Allegro(config=AllegroConfig), RobotEnvCfg, SamplingCfg
    elif env_name == "unitree_go2":
        if control_type == "sampling_method":
            return UnitreeGo2Env(config=UnitreeGo2EnvConfig), UnitreeGo2EnvConfig, SamplingCfg
    elif env_name == "unitree_h1":
        if control_type == "sampling_method":
            return UnitreeH1LocoEnv(config = UnitreeH1LocoEnvConfig), UnitreeH1LocoEnvConfig, SamplingCfg
    elif env_name in ["ant", "halfcheetah"]:
        if control_type == "sampling_method":
            return brax_envs.get_environment(env_name=env_name, backend="positional"), BaseEnvCfg, SamplingCfg
    else:
        raise ValueError(f"Unknown environment: {env_name}")
