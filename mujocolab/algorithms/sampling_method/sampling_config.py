from dataclasses import dataclass

## load config
@dataclass
class SamplingCfg:
    # exp
    seed: int = 0
    disable_recommended_params: bool = False
    not_render: bool = False
    # env
    env_name: str = (
        "ant"  # "humanoidstandup", "ant", "halfcheetah", "hopper", "walker2d", "car2d"
    )
    # diffusion
    Nsample: int = 2048  # number of samples
    Hsample: int = 50  # number of horizon
    Hnode : int = 50 # node number for control
    Nrefine: int = 100  # number of diffusion steps
    Ndiffuse: int =2 # number of diffusion steps
    Ndiffuse_init: int = 10 # diffusion step init
    traj_diffuse_factor : float = 0.9  # factor to scale the sigma of traj diffuse
    temp_sample: float = 0.1  # temperature for sampling
    beta0: float = 1e-4  # initial beta
    betaT: float = 1e-2  # final beta
    enable_demo: bool = False
    update_method : str = 'mppi'