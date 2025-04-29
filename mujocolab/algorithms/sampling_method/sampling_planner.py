import functools
import os
import jax
from jax import numpy as jnp
from jax import config
import tyro
from tqdm import tqdm
from matplotlib import pyplot as plt
import time
import mujocolab
import mujocolab.envs as envs
import mujocolab.utils as utils

from mujocolab.algorithms.sampling_method.sampling_config import SamplingCfg
import datetime
from jax_cosmo.scipy.interpolate import InterpolatedUnivariateSpline
import argparse
# Get the current date
current_datetime = datetime.datetime.now()

# Format it as day/month/year
formatted_datetime = current_datetime.strftime("%d-%m-%Y %H:%M")
# NOTE: enable this if you want higher precision
# config.update("jax_enable_x64", True)




@jax.jit
def softmax_update(weights, Y0s, sigma, mu_0t):
    mu_0tm1 = jnp.einsum("n,nij->ij", weights, Y0s)
    return mu_0tm1, sigma


@jax.jit
def cma_es_update(weights, Y0s, sigma, mu_0t):
    mu_0tm1 = jnp.einsum("n,nij->ij", weights, Y0s)
    Yerr = Y0s - mu_0t
    sigma = jnp.sqrt(jnp.einsum("n,nij->ij", weights, Yerr**2)).mean() * sigma
    sigma = jnp.maximum(sigma, 1e-3)
    return mu_0tm1, sigma


@jax.jit
def cem_update(weights, Y0s, sigma, mu_0t):
    # calculate 10 elites
    idx = jnp.argsort(weights)[::-1][:10]
    mu_0tm1 = jnp.mean(Y0s[idx], axis=0)
    return mu_0tm1, sigma

class Sampling_method:
    def __init__(self, config, env):
        self.env = env
        self.update_fn = {
            "mppi": softmax_update,
            "cma-es": cma_es_update,
            "cem": cem_update,
        }[config.update_method]
        ## setup env

        # recommended temperature for envs
        temp_recommend = {
            "ant": 0.1,
            "halfcheetah": 0.4,
            "hopper": 0.1,
            "humanoidstandup": 0.1,
            "humanoidrun": 0.1,
            "walker2d": 0.1,
            "pushT": 0.2,
        }
        Nrefine_recommend = {
            "pushT": 200,
            "humanoidrun": 300,
        }
        Nsample_recommend = {
            "humanoidrun": 8192,
        }
        Hsample_recommend = {
            "pushT": 40,
        }
        if not config.disable_recommended_params:
            config.temp_sample = temp_recommend.get(config.env_name, config.temp_sample)
            config.Nrefine = Nrefine_recommend.get(config.env_name, config.Nrefine)
            config.Nsample = Nsample_recommend.get(config.env_name, config.Nsample)
            config.Hsample = Hsample_recommend.get(config.env_name, config.Hsample)
            print(f"override temp_sample to {config.temp_sample}")
        self.Nrefine = config.Nrefine
        self.Nsample = config.Nsample
        self.Hsample = config.Hsample
        self.Hnode = config.Hnode
        self.temp_sample = config.temp_sample

        self.ctrl_dt = 0.02
        self.step_us = jnp.linspace(0, self.ctrl_dt * config.Hsample, config.Hsample + 1)
        self.step_nodes = jnp.linspace(0, self.ctrl_dt * config.Hsample, config.Hnode + 1)
        self.node_dt = self.ctrl_dt * (config.Hsample) / (config.Hnode)

        self.Nx = env.observation_size
        self.Nu = env.action_size

        ## diffusion coefficient
        self.betas = jnp.linspace(config.beta0, config.betaT, config.Nrefine)
        self.alphas = 1.0 - self.betas
        self.alphas_bar = jnp.cumprod(self.alphas)
        self.sigmas = jnp.sqrt(1 - self.alphas_bar)
        print(f"init sigma = {self.sigmas[-1]:.2e}")

        self.rollout_us = jax.jit(functools.partial(utils.simulation_utils.rollout_us, self.env.step))
        self.rollout_us_vmap = jax.jit(jax.vmap(self.rollout_us, in_axes=(None, 0)))

        self.node2u_vmap = jax.jit(
            jax.vmap(self.node2u, in_axes=(1), out_axes=(1))
        )  # process (horizon, node)
        self.u2node_vmap = jax.jit(jax.vmap(self.u2node, in_axes=(1), out_axes=(1)))
        self.node2u_vvmap = jax.jit(
            jax.vmap(self.node2u_vmap, in_axes=(0))
        )  # process (batch, horizon, node)
        self.u2node_vvmap = jax.jit(jax.vmap(self.u2node_vmap, in_axes=(0)))

        # YN = jnp.zeros([config.Hsample, Nu])

    @functools.partial(jax.jit, static_argnums=(0,))
    def node2u(self, nodes):
        spline = InterpolatedUnivariateSpline(self.step_nodes, nodes, k=2)
        us = spline(self.step_us)
        return us

    @functools.partial(jax.jit, static_argnums=(0,))
    def u2node(self, us):
        spline = InterpolatedUnivariateSpline(self.step_us, us, k=2)
        nodes = spline(self.step_nodes)
        return nodes


    @functools.partial(jax.jit, static_argnums=(0,))
    def reverse_once(self, state, rng, Ybar_i, sigma):
        # sample from q_i
        rng, Y0s_rng = jax.random.split(rng)
        eps_u = jax.random.normal(Y0s_rng, (self.Nsample, self.Hnode + 1, self.Nu))
     
        Y0s = eps_u * sigma + Ybar_i
        Y0s = jnp.clip(Y0s, -1.0, 1.0)

        us = self.node2u_vvmap(Y0s)
        rewss, pipeline_statess = self.rollout_us_vmap(state, us)
        rews =  rewss.mean(axis=-1)
        logp0 = (rews - rews.mean()) / rews.std() / self.temp_sample
        weights = jax.nn.softmax(logp0)

        qss = pipeline_statess.q
        qdss = pipeline_statess.qd
        xss = pipeline_statess.x.pos
        qbar = jnp.einsum("n,nij->ij", weights, qss)
        qdbar = jnp.einsum("n,nij->ij", weights, qdss)
        xbar = jnp.einsum("n,nijk->ijk", weights, xss)

        Ybar, sigma = self.update_fn(weights, Y0s, sigma, Ybar_i)

        info = {
            "rews": rews,
            "qbar": qbar,
            "qdbar": qdbar,
            "xbar": xbar,
            "new_noise_scale": sigma,
        }
        return rng, Ybar, info

    def reverse_scan(self, carry, sigmas):
        rng, state, Ybar_i = carry
        rng, Ybar, info = self.reverse_once(state, rng,Ybar_i, sigmas)
        return (rng, Ybar, state), info
    # run sampling
    def update(self, state, YN, rng):
        Yi = YN
        Ybars= []
        with tqdm(range(self.Nrefine - 1, 0, -1), desc="Sampling-Method") as pbar:
            for i in pbar:
                t0 = time.time()
                rng, Yi, info = self.reverse_once(state, rng, Yi, self.sigmas[i])
                Yi.block_until_ready()
                Ybars.append(Yi)
                # Update the progress bar's suffix to show the current reward
                freq = 1 / (time.time() - t0)
                rews=  info["rews"]
                pbar.set_postfix({"rew": f"{rews.mean():.2e}", "freq": f"{freq:.2f}"})
        return jnp.array(Ybars)

    @functools.partial(jax.jit, static_argnums=(0,))
    def shift(self, Y):
        u = self.node2u_vmap(Y)
        u = jnp.roll(u, -1, axis=0)
        u = u.at[-1].set(jnp.zeros(self.nu))
        Y = self.u2node_vmap(u)
        return Y

    def shift_Y_from_u(self, u, n_step):
        u = jnp.roll(u, -n_step, axis=0)
        u = u.at[-n_step:].set(jnp.zeros_like(u[-n_step:]))
        Y = self.u2node_vmap(u)
        return Y

if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # args = parser.parse_args()

    rng = jax.random.PRNGKey(seed=0)

    env, env_cfg, control_config = envs.get_env(config.env_name)
    step_env_jit = jax.jit(env.step)
    reset_env = jax.jit(env.reset)
    rng, rng_reset = jax.random.split(rng)
    state_init = reset_env(rng_reset)

    policy = Sampling_method(config=config, env=env)
    YN = jnp.zeros([config.Hnode + 1, env.action_size])
    Ybars = policy.update(state_init, YN, rng)
    if not config.not_render:
        path = f"{mujocolab.__path__[0]}/../results/{config.env_name}"
        if not os.path.exists(path):
            os.makedirs(path)
        jnp.save(f"{path}/mu_0ts.npy", Ybars)
        render_us = functools.partial(
                utils.simulation_utils.render_us,
                step_env_jit,
                env.sys.tree_replace({"opt.timestep": env.dt}),
            )
        webpage = render_us(state_init, Ybars[-1])
        date = datetime.datetime.now()
        with open(f"{path}/rollout_{date}.html", "w") as f:
            f.write(webpage)
    

