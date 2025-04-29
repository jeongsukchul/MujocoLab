import os
import time
from multiprocessing import shared_memory
from dataclasses import dataclass
import importlib
import sys
import argparse
import yaml
import argparse
import numpy as np
import matplotlib.pyplot as plt
import scienceplots
import mujoco
import mujoco.viewer
import mujocolab
from mujocolab.envs.env_config.base_env_config import BaseEnvCfg
from mujocolab.algorithms.sampling_method.sampling_config import SamplingCfg
# from mujocolab.utils.io_utils import (
#     load_dataclass_from_dict,
# )
from mujocolab.utils.simulation_utils import get_model_path
from mujocolab.envs import get_env
plt.style.use(["science"])
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
@dataclass
class MujocoSimCfg:
    robot_name: str 
    scene_name: str
    plot: bool = False
    record: bool = False
    real_time_factor: float = 1.0
    sim_dt: float = 0.05
    sync_mode: bool = False


class MujocoSim:
    def __init__(
        self,
        env_config: BaseEnvCfg,
        control_cfg: SamplingCfg,
    ):
        # control related
        self.plot = MujocoSimCfg.plot
        self.record = MujocoSimCfg.record
        self.real_time_factor = MujocoSimCfg.real_time_factor
        self.sim_dt = MujocoSimCfg.sim_dt
        self.sync_mode = MujocoSimCfg.sync_mode

        self.data = []
        self.ctrl_dt = env_config.dt
        self.n_acts = control_cfg.Hsample + 1
        self.n_frame = int(self.ctrl_dt / self.sim_dt)
        self.t = 0.0
        self.leg_control = env_config.leg_control
        if env_config.scene_name is None:
            model_path = f"{mujocolab.__path__[0]}/assets/{env_config.robot_name}.xml"
        else:
            model_path  =  get_model_path(env_config.robot_name, env_config.scene_name)
        self.mj_model = mujoco.MjModel.from_xml_path(model_path)
        self.mj_model.opt.timestep = self.sim_dt
        self.mj_data = mujoco.MjData(self.mj_model)
        self.q_history = np.zeros((self.n_acts, self.mj_model.nu))
        self.qref_history = np.zeros((self.n_acts, self.mj_model.nu))
        self.n_plot_joint = 4

        # mujoco setup
        mujoco.mj_resetDataKeyframe(self.mj_model, self.mj_data, 0)
        mujoco.mj_forward(self.mj_model, self.mj_data)

        # parameters
        self.Nx = self.mj_model.nq + self.mj_model.nv
        self.Nu = self.mj_model.nu

        # get home keyframe
        try: 
            self.default_q = self.mj_model.keyframe("home").qpos
            self.default_u = self.mj_model.keyframe("home").ctrl
        except KeyError:
            self.default_q = np.zeros(self.mj_model.nq)
            self.default_u = np.zeros(self.mj_model.nu)
        # communication setup
        # publisher
        self.time_shm = shared_memory.SharedMemory(
            name="time_shm", create=True, size=32
        )
        self.time_shared = np.ndarray(1, dtype=np.float32, buffer=self.time_shm.buf)
        self.time_shared[0] = 0.0
        self.state_shm = shared_memory.SharedMemory(
            name="state_shm", create=True, size=self.Nx * 32
        )
        self.state_shared = np.ndarray(
            (self.Nx,), dtype=np.float32, buffer=self.state_shm.buf
        )
        # listener
        self.acts_shm = shared_memory.SharedMemory(
            name="acts_shm", create=True, size=self.n_acts * self.Nu * 32
        )
        self.acts_shared = np.ndarray(
            (self.n_acts, self.mj_model.nu), dtype=np.float32, buffer=self.acts_shm.buf
        )
        self.acts_shared[:] = self.default_u
        self.refs_shm = shared_memory.SharedMemory(
            name="refs_shm", create=True, size=self.n_acts * self.Nu * 3 * 32
        )
        self.refs_shared = np.ndarray(
            (self.n_acts, self.Nu, 3), dtype=np.float32, buffer=self.refs_shm.buf
        )
        self.refs_shared[:] = 0.0
        self.plan_time_shm = shared_memory.SharedMemory(
            name="plan_time_shm", create=True, size=32
        )
        self.plan_time_shared = np.ndarray(
            1, dtype=np.float32, buffer=self.plan_time_shm.buf
        )
        self.plan_time_shared[0] = -self.ctrl_dt

        self.tau_shm = shared_memory.SharedMemory(
            name="tau_shm", create=True, size=self.n_acts * self.Nu * 32
        )
        self.tau_shared = np.ndarray(
            (self.n_acts, self.mj_model.nu), dtype=np.float32, buffer=self.tau_shm.buf
        )
        self.reset_shm = shared_memory.SharedMemory(name="reset_shm", create=True, size=32)
        self.reset_shared = np.ndarray(1, dtype=np.int32, buffer=self.reset_shm.buf)
        self.reset_shared[0] = 0  # initially no reset

    def main_loop(self):
        if self.plot:
            fig, axs = plt.subplots(self.n_plot_joint, 1, figsize=(12, 12))
            # plot history
            handles = []
            handles_ref = []
            # colors for each joint with rainbow
            colors = plt.cm.rainbow(np.linspace(0, 1, self.n_plot_joint))
            for i in range(self.n_plot_joint):
                handles.append(
                    axs[i].plot(
                        self.q_history[:, i],
                        color=colors[i],
                    )[0]
                )
                handles_ref.append(
                    axs[i].plot(
                        self.qref_history[:, i],
                        color=colors[i],
                        linestyle="--",
                    )[0]
                )
                # set ylim to [-0.5, 0.5]
                axs[i].set_ylim(
                    -1.0 + self.default_q[i + 7], 1.0 + self.default_q[i + 7]
                )
                axs[i].set_xlabel("Time (s)")
                axs[i].set_ylabel(f"Joint {i+1} Position")
            # show figure
            plt.show(block=False)

        viewer = mujoco.viewer.launch_passive(
            self.mj_model, self.mj_data, show_left_ui=False, show_right_ui=False
        )
        # Check for reset request
        if self.reset_shared[0] == 1:
            print("[Simulator] Resetting environment...")
            mujoco.mj_resetDataKeyframe(self.mj_model, self.mj_data, 0)  # reset all states
            mujoco.mj_forward(self.mj_model, self.mj_data)    # recompute after reset
            self.t = 0.0
            self.plan_time_shared[0] = -self.ctrl_dt  # optional: clear planning time
            self.reset_shared[0] = 0  # acknowledge reset complet
        cnt = 0
        viewer.user_scn.ngeom = 0
        for i in range(self.n_acts - 1):
            # iterate over all geoms
            for j in range(self.mj_model.nu):
                color = np.array(
                    [1.0 * i / (self.n_acts - 1), 1.0 * j / self.mj_model.nu, 0.0, 1.0]
                )
                mujoco.mjv_initGeom(
                    viewer.user_scn.geoms[cnt],
                    type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                    size=np.zeros(3),
                    rgba=color,
                    pos=self.refs_shared[i, j, :],
                    mat=np.eye(3).flatten(),
                )
                cnt += 1
        viewer.user_scn.ngeom = cnt
        viewer.sync()
        while True:
            if self.plot:
                # plot self.acts_shared
                for j in range(self.n_plot_joint):
                    # update plot
                    handles[j].set_ydata(self.acts_shared[:, j])
                    handles_ref[j].set_ydata(self.qref_history[:, j])
                plt.pause(0.001)
            # update geoms according to the reference
            for i in range(self.n_acts - 1):
                for j in range(self.mj_model.nu):
                    r0 = self.refs_shared[i, j, :]
                    r1 = self.refs_shared[i + 1, j, :]
                    mujoco.mjv_connector(
                        viewer.user_scn.geoms[i * self.mj_model.nu + j],
                        mujoco.mjtGeom.mjGEOM_CAPSULE,
                        0.02,
                        r0,
                        r1,
                    )
            if self.sync_mode:
                while self.t <= (self.plan_time_shared[0] + self.ctrl_dt):
                    if self.leg_control == "position":
                        self.mj_data.ctrl = self.acts_shared[0]
                    elif self.leg_control == "torque":
                        self.mj_data.ctrl = self.tau_shared[0]
                    if self.record:
                        self.data.append(
                            np.concatenate(
                                [
                                    [self.t],
                                    self.mj_data.qpos,
                                    self.mj_data.qvel,
                                    self.mj_data.ctrl,
                                ]
                            )
                        )
                    mujoco.mj_step(self.mj_model, self.mj_data)
                    self.t += self.sim_dt
                    # publish new state
                    q = self.mj_data.qpos
                    qd = self.mj_data.qvel
                    state = np.concatenate([q, qd])
                    self.time_shared[:] = self.t
                    self.state_shared[:] = state
                self.q_history = np.roll(self.q_history, -1, axis=0)
                self.q_history[-1, :] = q[7:]
                self.qref_history = np.roll(self.qref_history, -1, axis=0)
                self.qref_history[-1, :] = self.mj_data.ctrl
                viewer.sync()
            else:
                t0 = time.time()
                if self.plan_time_shared[0] < 0.0:
                    time.sleep(0.01)
                    continue
                delta_time = self.t - self.plan_time_shared[0]
                delta_step = int(delta_time / self.ctrl_dt)
                if delta_time > self.ctrl_dt / self.real_time_factor:
                    print(f"[WARN] Delayed by {delta_time*1000.0:.1f} ms")
                if delta_step >= self.n_acts or delta_step < 0:
                    delta_step = self.n_acts - 1

                if self.leg_control == "position":
                    self.mj_data.ctrl = self.acts_shared[delta_step]
                elif self.leg_control == "torque":
                    self.mj_data.ctrl = self.tau_shared[delta_step]
                if self.record:
                    self.data.append(
                        np.concatenate(
                            [
                                [self.t],
                                self.mj_data.qpos,
                                self.mj_data.qvel,
                                self.mj_data.ctrl,
                            ]
                        )
                    )
                mujoco.mj_step(self.mj_model, self.mj_data)
                self.t += self.sim_dt
                q = self.mj_data.qpos
                qd = self.mj_data.qvel
                state = np.concatenate([q, qd])

                # publish new state
                self.time_shared[:] = self.t
                self.state_shared[:] = state

                self.q_history = np.roll(self.q_history, -1, axis=0)
                self.q_history[-1, :] = q[7:]
                self.qref_history = np.roll(self.qref_history, -1, axis=0)
                self.qref_history[-1, :] = self.mj_data.ctrl
                viewer.sync()
                t1 = time.time()
                duration = t1 - t0
                if duration < self.sim_dt / self.real_time_factor:
                    time.sleep((self.sim_dt / self.real_time_factor - duration))
                else:
                    print("[WARN] Sim loop overruns")

    def close(self):
        self.time_shm.close()
        self.time_shm.unlink()
        self.state_shm.close()
        self.state_shm.unlink()
        self.acts_shm.close()
        self.acts_shm.unlink()
        self.plan_time_shm.close()
        self.plan_time_shm.unlink()
        self.refs_shm.close()
        self.refs_shm.unlink()
        self.tau_shm.close()
        self.tau_shm.unlink()
        self.reset_shm.close()
        self.reset_shm.unlink()

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
    mujoco_env = MujocoSim(env_cfg, control_cfg)

    try:
        mujoco_env.main_loop()
    except KeyboardInterrupt:
        pass
    finally:
        if mujoco_env.record:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            data = np.array(mujoco_env.data)
            output_dir = os.path.join(
                control_cfg.output_dir,
                f"sim_{control_cfg.env_name}_{env_cfg.task_name}_{timestamp}",
            )
            os.makedirs(output_dir)
            np.save(os.path.join(output_dir, "states"), data)

        mujoco_env.close()


if __name__ == "__main__":
    main()
