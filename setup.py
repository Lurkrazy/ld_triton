from setuptools import setup, find_packages

setup(
    name='ld_triton',
    version='0.1',
    packages=find_packages(),
    author='Shuliang Li',
    license='MIT',
    install_requires=[
        'torch>=2.3',
        'triton>=3.0',
    ],
    extras_require={
        'test': [
            'pytest', 
        ],
    },
    python_requires='>=3.9',
)