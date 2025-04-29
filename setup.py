from setuptools import setup, find_packages

# Read the requirements from the requirements.txt file
with open('requirements.txt', 'r') as f:
    requirements = f.read().splitlines()


setup(name='mujocolab',
    author="Sukchul Jeong",
    author_email="tjrcjf410@snu.ac.kr",
    packages=find_packages(include="mujocolab"),
    version='0.0.1',
    install_requires=[
        'gym', 
        'pandas', 
        'seaborn', 
        'matplotlib', 
        'imageio',
        'control', 
        'tqdm', 
        'tyro', 
        'meshcat', 
        'sympy', 
        'gymnax',
        'jax[cuda12]',
        'jax-cosmo', 
        'distrax', 
        'gputil', 
        'jaxopt',
        'brax'
        ]
)