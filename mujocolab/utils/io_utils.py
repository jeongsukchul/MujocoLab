import importlib.resources
import os

# def get_model_path(robot_name, model_name):
#     with importlib.resources.path(f"mujocolab.models.{robot_name}", model_name) as path:
#         print("path", path)
#         return path


# def get_example_path(example_name):
#     with importlib.resources.path(f"mujocolab.examples", example_name) as path:
#         return path


def load_dataclass_from_dict(dataclass, data_dict, convert_list_to_array=False):
    keys = dataclass.__dataclass_fields__.keys() & data_dict.keys()
    kwargs = {key: data_dict[key] for key in keys}
    if convert_list_to_array:
        import jax.numpy as jnp

        for key, value in kwargs.items():
            if isinstance(value, list):
                kwargs[key] = jnp.array(value)
    return dataclass(**kwargs)
