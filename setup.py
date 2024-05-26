from setuptools import setup, find_packages


setup(
    name='g2rl',
    version='0.1.0',
    author='julia-bel',
    license='MIT',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    description='Implementation of G2RL in the POGEMA environment',
    install_requires=[
        "numpy",
        "pandas",
        "torch",
    ],
    packages=find_packages(),
    python_requires='>=3.9',
)
