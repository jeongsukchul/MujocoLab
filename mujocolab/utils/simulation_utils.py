import jax
from brax.io import html
import importlib.resources
import mujocolab
# evaluate the diffused uss
def eval_us(step_env, state, us):
    def step(state, u):
        state = step_env(state, u)
        return state, state.reward

    _, rews = jax.lax.scan(step, state, us)
    return rews

def rollout_us(step_env, state, us):
    def step(state, u):
        state = step_env(state, u)
        return state, (state.reward, state.pipeline_state)

    _, (rews, pipline_states) = jax.lax.scan(step, state, us)
    return rews, pipline_states


def render_us(step_env, sys, state, us):
    rollout = []
    rew_sum = 0.0
    Hsample = us.shape[0]
    for i in range(Hsample):
        rollout.append(state.pipeline_state)
        state = step_env(state, us[i])
        rew_sum += state.reward
    # rew_mean = rew_sum / (Hsample)
    # print(f"evaluated reward mean: {rew_mean:.2e}")
    return html.render(sys, rollout)

def get_model_path(robot_name, scene_name):
    path = f"{mujocolab.__path__[0]}/assets/{robot_name}/{scene_name}"
    return path