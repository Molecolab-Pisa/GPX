from setuptools import find_packages, setup

setup(
    name="GPX",
    version="0.1.0",
    url="https://github.com/ecignoni/GPX",
    author="Edoardo Cignoni, Amanda Arcidiacono, Patrizia Mazzeo",
    author_email="edoardo.cignoni96@gmail.com",
    description=open("README.md").read(),
    packages=find_packages(),
    install_requires=["typing_extensions", "tqdm", "optax", "tabulate", "numpy"],
)
