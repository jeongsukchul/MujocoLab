import jax
import time
import jax.numpy as jnp
import numpy as np
from multiprocessing import shared_memory
from mujoco import mjx
import keyboard

from mujocolab.algorithms.sampling_method.sampling_config import SamplingCfg
from mujocolab.algorithms.sampling_method.sampling_planner import Sampling_method
from mujocolab.envs.env_config.base_env_config import BaseEnvCfg
from mujocolab.envs.base_env import BaseEnv
from mujocolab.algorithms.control_publisher import ControlPublisher
import brax.envs as brax_envs
from brax.envs.base import Env as BraxEnv
from brax.envs.base import State
from brax.mjx.base import State as MjxState
from brax.mjx.pipeline import _reformat_contact
from jax_cosmo.scipy.interpolate import InterpolatedUnivariateSpline
from brax.base import Contact, Motion, System, Transform
from jax_cosmo.scipy.interpolate import InterpolatedUnivariateSpline

def pipeline_init(
    sys: System,
    q: jax.Array,
    qd: jax.Array,
) -> MjxState:
    data = mjx.make_data(sys)
    data = data.replace(qpos=q, qvel=qd)

    q, qd = data.qpos, data.qvel
    x = Transform(pos=data.xpos[1:], rot=data.xquat[1:])
    cvel = Motion(vel=data.cvel[1:, 3:], ang=data.cvel[1:, :3])
    offset = data.xpos[1:, :] - data.subtree_com[sys.body_rootid[1:]]
    offset = Transform.create(pos=offset)
    xd = offset.vmap().do(cvel)

    data = _reformat_contact(sys, data)
    return MjxState(q=q, qd=qd, x=x, xd=xd, **data.__dict__)

class SamplingPublisher(ControlPublisher):

    def __init__(
            self, env: BaseEnv, EnvCfg: BaseEnvCfg, SamplingCfg: SamplingCfg
    ):
        self.env = env
        self.env_config = EnvCfg
        self.sampling_config = SamplingCfg

        self.sampling_planner = Sampling_method(SamplingCfg, self.env)
        self.rng = jax.random.PRNGKey(SamplingCfg.seed)

        self.pipeline_init_jit = jax.jit(pipeline_init)
        self.shift_vmap = jax.jit(jax.vmap(self.shift, in_axes=(1, None), out_axes=1))

        # control parameters
        self.Y = jnp.zeros([self.sampling_config.Hnode + 1, self.env.sys.mj_model.nu])
        self.ctrl_dt = self.env_config.dt

        # parameters
        self.timer_period = self.env_config.dt  # seconds
        self.n_acts = self.sampling_config.Hsample + 1  # action buffer size
        self.nx = self.env.sys.mj_model.nq + self.env.sys.mj_model.nv
        self.nu = self.env.sys.mj_model.nu

        try: 
            self.default_q = self.env.sys.mj_model.keyframe("home").qpos
            self.default_u = self.env.sys.mj_model.keyframe("home").ctrl
        except KeyError:
            self.default_q = np.zeros(self.mj_model.nq)
            self.default_u = np.zeros(self.mj_model.nu)
        # publisher
        self.acts_shm = shared_memory.SharedMemory(
            name="acts_shm", create=False, size=self.n_acts * self.nu * 32
        )
        self.acts_shared = np.ndarray(
            (self.n_acts, self.nu), dtype=np.float32, buffer=self.acts_shm.buf
        )
        self.acts_shared[:] = self.default_u
        self.refs_shm = shared_memory.SharedMemory(
            name="refs_shm", create=False, size=self.n_acts * self.env.sys.nu * 3 * 32
        )
        self.refs_shared = np.ndarray(
            (self.n_acts, self.env.sys.nu, 3),
            dtype=np.float32,
            buffer=self.refs_shm.buf,
        )
        self.refs_shared[:] = 1.0
        self.plan_time_shm = shared_memory.SharedMemory(
            name="plan_time_shm", create=False, size=32
        )
        self.plan_time_shared = np.ndarray(
            1, dtype=np.float32, buffer=self.plan_time_shm.buf
        )
        self.plan_time_shared[0] = -0.02
        # listerner
        self.time_shm = shared_memory.SharedMemory(
            name="time_shm", create=False, size=32
        )
        self.time_shared = np.ndarray(1, dtype=np.float32, buffer=self.time_shm.buf)
        self.time_shared[0] = 0.0
        self.state_shm = shared_memory.SharedMemory(
            name="state_shm", create=False, size=self.nx * 32
        )
        self.state_shared = np.ndarray(
            (self.nx,), dtype=np.float32, buffer=self.state_shm.buf
        )
        self.state_shared[: self.default_q.shape[0]] = self.default_q

        self.tau_shm = shared_memory.SharedMemory(
            name="tau_shm", create=False, size=self.n_acts * self.nu * 32
        )
        self.tau_shared = np.ndarray(
            (self.n_acts, self.nu), dtype=np.float32, buffer=self.tau_shm.buf
        )
        self.reset_shm = shared_memory.SharedMemory(name="reset_shm")
        self.reset_shared = np.ndarray(1, dtype=np.int32, buffer=self.reset_shm.buf)

    def shift(self, x, shift_time):
        spline = InterpolatedUnivariateSpline(self.sampling_planner.step_nodes, x, k=2)
        x_new = spline(self.sampling_planner.step_nodes + shift_time)
        return x_new

    def init_mjx_state(self, q, qd, t):
        state = self.env.reset(jax.random.PRNGKey(0))
        pipeline_state = self.pipeline_init_jit(self.env.sys, q, qd)
        obs = self.env._get_obs(pipeline_state, state.info)
        state = state.replace(pipeline_state=pipeline_state, obs=obs)
        return state

    # @partial(jax.jit, static_argnums=(0,))
    def update_mjx_state(self, state, q, qd, t):
        pipeline_state = state.pipeline_state.replace(qpos=q, qvel=qd)
        step = int(t / self.ctrl_dt)
        info = state.info
        info["step"] = step
        state = state.replace(pipeline_state=pipeline_state, info=info)
        return state
    
    def main_loop(self):
        
        def reverse_scan(rng_Y0_state, factor):
            rng, Y0, state = rng_Y0_state
            rng, Y0, info = self.sampling_planner.reverse_once(state, rng, Y0, factor)
            return (rng, Y0, state), info

        last_plan_time = self.time_shared[0]
        state = self.init_mjx_state(
            self.state_shared[: self.env.sys.mj_model.nq].copy(),
            self.state_shared[self.env.sys.mj_model.nq :].copy(),
            last_plan_time.copy(),
        )

        first_time = True
        while True:
            t0 = time.time()
            # get state
            plan_time = self.time_shared[0]
            state = self.update_mjx_state(
                state,
                self.state_shared[: self.env.sys.mj_model.nq],
                self.state_shared[self.env.sys.mj_model.nq :],
                plan_time,
            )
            # shift Y
            shift_time = plan_time - last_plan_time
            if shift_time > self.ctrl_dt + 1e-3:
                print(f"[WRAN] sim overtime {(shift_time-self.ctrl_dt)*1000:.1f} ms")
            if shift_time > self.ctrl_dt * self.n_acts:
                print(
                    f"[WARN] long time unplanned {shift_time*1000:.1f} ms, reset control"
                )
                self.Y = self.Y * 0.0
            else:
                self.Y = self.shift_vmap(self.Y, shift_time)
            # run planner
            n_diffuse = self.sampling_config.Nrefine
            if first_time:
                print("Performing JIT on DIAL-MPC")
                n_diffuse = self.sampling_config.Ndiffuse_init
                first_time = False
                traj_diffuse_factors = (
                    self.sampling_config.traj_diffuse_factor
                    ** (jnp.arange(n_diffuse))[:, None]
                )
                (self.rng, self.Y, _), info = jax.lax.scan(
                    reverse_scan, (self.rng, self.Y, state), traj_diffuse_factors
                )
                n_diffuse = self.sampling_config.Nrefine
            traj_diffuse_factors = (
                self.sampling_config.traj_diffuse_factor ** (jnp.arange(n_diffuse))[:, None]
            )
            (self.rng, self.Y, _), info = jax.lax.scan(
                reverse_scan, (self.rng, self.Y, state), traj_diffuse_factors
            )
            # use position control
            actual_joint_targets = info["qbar"][:, 7:]
            x_targets = info["xbar"][-1, :, 1:, :3]
            # convert plan to control
            us = self.sampling_planner.node2u_vmap(self.Y)
            # unnormalize control
            joint_targets = self.env.act2joint(us)
            taus = self.env.act2tau(us, state.pipeline_state)
            # send control
            self.acts_shared[: joint_targets.shape[0], :] = joint_targets
            self.tau_shared[: taus.shape[0], :] = taus
            self.plan_time_shared[0] = plan_time
            self.refs_shared[:, :, :] = x_targets[: self.refs_shared.shape[0], :, :]
            # record time
            last_plan_time = plan_time
            if time.time() - t0 > self.ctrl_dt:
                print(f"[WRAN] real overtime {(time.time()-t0)*1000:.1f} ms")
            # if keyboard.is_pressed('r'):
            #     print("[Controller] Reset requested by keypress.")
            #     self.reset_shared[0] = 1
