from setuptools import setup, find_packages

setup(
    name="polaris",
    version="2.0.0",
    author="Your Name",
    description="POLARIS: Partially Observable Learning with Active Reinforcement In Social Environments",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/polaris",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.8.0",
        "numpy>=1.20.0",
        "matplotlib>=3.4.0",
        "networkx>=2.6.0",
        "scikit-learn>=1.0.0",
        "tqdm>=4.60.0",
        "scipy>=1.7.0",
    ],
    extras_require={
        "gnn": [
            "torch-geometric>=2.0.0",
            "torch-scatter>=2.0.9",
            "torch-sparse>=0.6.13",
            "torch-cluster>=1.6.0",
            "torch-spline-conv>=1.2.1",
        ],
    },
)