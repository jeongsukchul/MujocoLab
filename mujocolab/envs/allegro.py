from brax import base
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
from brax.base import System
from etils import epath
import jax
from jax import numpy as jnp
from functools import partial
import numpy as np
import mujocolab
from mujocolab.envs.base_env import BaseEnv, BaseEnvCfg
from typing import Any, Dict, Sequence, Tuple, Union, List
from mujocolab.utils.simulation_utils import get_model_path
class AllegroConfig(BaseEnvCfg):
    kp: Union[float, jax.Array] = 30.0
    kd: Union[float, jax.Array] = 0.0
    robot_name : str = "allegro"
    scene_name : str = "scene.xml"
class Allegro(BaseEnv):
    def __init__(self, config: AllegroConfig):
        super().__init__(config)
        # sys = self.make_system(AllegroConfig)

        n_frames = 2
        self.joint_torque_range = sys.actuator_ctrlrange
        self.joint_range = self.physical_joint_range
        self.physical_joint_range = self.sys.jnt_range[1:]

        self.desired_angle = np.random.uniform(-np.pi/3, np.pi/3)
        self.desired_vel = np.random.uniform(-1, 1)
        

        self._init_q = jnp.array(self.sys.mj_model.keyframe("home").qpos)
        self._default_pose = self.sys.mj_model.keyframe("home").qpos[7:]
        self._nv = self.sys.nv
        self._nq = self.sys.nq
        print(" nq nv init q default pose", self._nq, self._nv, self._init_q, self._default_pose)

    def make_system(self, config: AllegroConfig) -> System:
        model_path = get_model_path("allegro", "scene.xml")
        print("model path", model_path)

        sys = mjcf.load(model_path)
        sys = sys.tree_replace({"opt.timestep": config.timestep})
        return sys
    

    def reset(self, rng: jax.Array) -> State:
        """Resets the environment to an initial state."""
        rng, rng1, rng2 = jax.random.split(rng, 3)

        pipeline_state = self.pipeline_init(self._init_q, jnp.zeros(self._nv))


        state_info = {
            "rng" : rng,
            "ori_goal" : jnp.array([])
        }

        obs = self._get_obs(pipeline_state, state_info)
        reward, done = jnp.zeros(2)
        metrics = {}

        return State(pipeline_state, obs, reward, done, metrics)

    def step(self, state: State, action: jax.Array) -> State:
        """Run one timestep of the environment's dynamics."""
        rng, cmd_rng = jax.random.split(state.info["rng"], 2)
        joint_targets = self.act2joint(action)
        if self._config.leg_control == "position":
            ctrl = joint_targets
        elif self._config.leg_control == "torque":
            ctrl = self.act2tau(action, state.pipeline_state)
        pipeline_state = self.pipeline_step(state.pipeline_state, ctrl)
        
        obs = self._get_obs(pipeline_state, state.info)
        
        #reward
        reward = 1.0 - jnp.abs(pipeline_state.qd[2] -self.desired_vel) - jnp.abs(pipeline_state.q[2] - self.desired_angle)/(2*jnp.pi)
        done = 0.0
        return state.replace(
            pipeline_state=pipeline_state, obs=obs, reward=reward, done=done
        )
    @property
    def action_size(self):
        return 16
    @partial(jax.jit, static_argnums=(0,))
    def act2joint(self, act: jax.Array) -> jax.Array:
        act_normalized = (
            act * self._config.action_scale + 1.0
        ) / 2.0  # normalize to [0, 1]
        joint_targets = self.joint_range[:, 0] + act_normalized * (
            self.joint_range[:, 1] - self.joint_range[:, 0]
        )  # scale to joint range
        joint_targets = jnp.clip(
            joint_targets,
            self.physical_joint_range[:, 0],
            self.physical_joint_range[:, 1],
        )
        return joint_targets

    @partial(jax.jit, static_argnums=(0,))
    def act2tau(self, act: jax.Array, pipline_state) -> jax.Array:
        joint_target = self.act2joint(act)

        q = pipline_state.qpos[7:]
        q = q[: len(joint_target)]
        qd = pipline_state.qvel[6:]
        qd = qd[: len(joint_target)]
        q_err = joint_target - q
        tau = self._config.kp * q_err - self._config.kd * qd

        tau = jnp.clip(
            tau, self.joint_torque_range[:, 0], self.joint_torque_range[:, 1]
        )
        return tau

    def _get_obs(self, pipeline_state: base.State) -> jax.Array:
        """Observe cartpole body position and velocities."""
        return jnp.concatenate([pipeline_state.q, pipeline_state.qd])
