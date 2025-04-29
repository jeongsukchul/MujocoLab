from brax import base
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
from etils import epath
import jax
from jax import numpy as jp
from functools import partial
import numpy as np
import mujocolab


class OneFinger(PipelineEnv):
    def __init__(self, backend="positional", **kwargs):
        sys = mjcf.load(f"{mujocolab.__path__[0]}/assets/finger.xml")

        n_frames = 2

        if backend in ["spring", "positional"]:
            # sys = sys.replace(dt=0.005)
            n_frames = 4

        kwargs["n_frames"] = kwargs.get("n_frames", n_frames)
        self.joint_torque_range = sys.actuator_ctrlrange
        self.desired_angle = np.random.uniform(-np.pi/3, np.pi/3)
        self.desired_vel = np.random.uniform(-1, 1)
        

        super().__init__(sys=sys, backend=backend, **kwargs)

    def reset(self, rng: jax.Array) -> State:
        """Resets the environment to an initial state."""
        rng, rng1, rng2 = jax.random.split(rng, 3)

        q = self.sys.init_q + jax.random.uniform(
            rng1, (self.sys.q_size(),), minval=-0.01, maxval=0.01
        ) + jp.array([jp.pi/2, -jp.pi/2, jp.pi, self.desired_angle])
        qd = jax.random.uniform(rng2, (self.sys.qd_size(),), minval=-0.01, maxval=0.01) + jp.array([0, 0, 10.0, self.desired_vel])
        pipeline_state = self.pipeline_init(q, qd)
        obs = self._get_obs(pipeline_state)
        reward, done = jp.zeros(2)
        metrics = {}

        return State(pipeline_state, obs, reward, done, metrics)

    def step(self, state: State, action: jax.Array) -> State:
        """Run one timestep of the environment's dynamics."""
        tau = self.act2tau(action)
        pipeline_state = self.pipeline_step(state.pipeline_state, tau)
        obs = self._get_obs(pipeline_state)
        reward = 1.0 - jp.abs(pipeline_state.qd[2] -self.desired_vel) - jp.abs(pipeline_state.q[2] - self.desired_angle)/(2*jp.pi)
        done = 0.0
        return state.replace(
            pipeline_state=pipeline_state, obs=obs, reward=reward, done=done
        )
    @property
    def action_size(self):
        return 2
    @partial(jax.jit, static_argnums=(0,))
    def act2tau(self, act):
        act_normalized = (
            act + 1.0
        ) / 2.0  # normalize to [0, 1]
        tau = self.joint_torque_range[:, 0] + act_normalized * (
            self.joint_torque_range[:, 1] - self.joint_torque_range[:, 0]
        )  # scale to joint range
        return tau

    def _get_obs(self, pipeline_state: base.State) -> jax.Array:
        """Observe cartpole body position and velocities."""
        return jp.concatenate([pipeline_state.q, pipeline_state.qd])
