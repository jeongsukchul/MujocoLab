
import jax
import brax.envs as brax_envs
import argparse
from mujocolab.envs import get_env
from mujocolab.algorithms.control_publisher import BaseControlConfig
from mujocolab.algorithms.sampling_method.sampling_config import SamplingCfg
from mujocolab.algorithms.sampling_method.sampling_control import SamplingPublisher


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env_name",
        type=str,
        default="pushT",
        help="Env to Control",
    )
    parser.add_argument(
        "--control_type",
        type=str,
        default="sampling_method",
        help="Env to Control",
    )
    args = parser.parse_args(args)

    env, env_cfg, control_cfg = get_env(args.env_name, args.control_type)

    if env_cfg.robot_name is None:
        env_cfg.robot_name = args.env_name
    control_publisher = SamplingPublisher(env, env_cfg, control_cfg)

    try:
        control_publisher.main_loop()
    except KeyboardInterrupt:
        pass
if __name__ == "__main__":
    main()
