from setuptools import setup, find_packages

setup(
    name="ia753-project",
    version="0.1.0",
    description="IA753 Final Work Project",
    author="Mateus Schneider Castilhos and Renan Ribeiro Machado",
    package_dir={"": "src"},
    packages=find_packages("src"),
    python_requires=">=3.13,<3.15",
    install_requires=[
        "numpy>=1.24,<2.0",
        "scipy>=1.10,<2.0",
        "pandas>=2.0,<3.0",
        "scikit-learn>=1.3,<2.0",
        "matplotlib>=3.8,<4.0",
        "seaborn>=0.13,<1.0",
        "ipykernel>=6.25,<8.0",
        "ipython>=8.15,<9.0",
        "ipywidgets>=8.1,<9.0",
    ],
    extras_require={
        "dev": ["pytest>=7.0,<9.0", "black>=24.0,<26.0", "mypy>=1.0,<2.0"],
    },
)
