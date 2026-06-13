from setuptools import setup, find_packages

setup(
    name="ia753-project",
    version="0.1.0",
    description="IA753 Final Work Project",
    author="Your Name",
    package_dir={"": "src"},
    packages=find_packages("src"),
    python_requires=">=3.13",
    install_requires=[],
    extras_require={
        "dev": ["pytest>=7.0", "black>=24.0", "mypy>=1.0"],
    },
)
