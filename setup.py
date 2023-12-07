from setuptools import find_packages, setup

setup(
    name="GPX",
    version="0.1.0",
    url="https://github.com/Molecolab-Pisa/GPX",
    author="Edoardo Cignoni, Patrizia Mazzeo, Amanda Arcidiacono, Lorenzo Cupellini, Benedetta Mennucci",  # noqa
    author_email="edoardo.cignoni96@gmail.com, mazzeo.patrizia.1998@gmail.com, amy.arci@gmail.com",  # noqa
    description=open("README.md").read(),
    packages=find_packages(),
    install_requires=["typing_extensions", "tqdm", "tabulate", "numpy"],
)
