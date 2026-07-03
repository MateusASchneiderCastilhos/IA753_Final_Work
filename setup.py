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
        "mne>=1.12,<2.0",
        "numpy>=2.0,<3.0",
        "scipy>=1.13,<2.0",
        "pandas>=3.0,<4.0",
        "scikit-learn>=1.4,<2.0",
        "matplotlib>=3.9,<4.0",
        "seaborn>=0.13,<1.0",
    ],
    extras_require={
        "notebook": ["ipykernel>=6.25,<8.0", "ipywidgets>=8.1,<9.0"],
        "dev": ["pytest>=7.0,<9.0", "black>=24.0,<26.0", "mypy>=1.0,<2.0"],
    },
)
