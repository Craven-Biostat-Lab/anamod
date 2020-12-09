"""anamod package definition and install configuration"""

from setuptools import find_packages, setup

# pylint: disable = invalid-name

with open("README.rst") as readme_file:
    readme = readme_file.read()

with open("docs/changelog.rst") as changelog_file:
    changelog = changelog_file.read()

setup(
    author="Akshay Sood",
    author_email="sood.iitd@gmail.com",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: Implementation :: CPython",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    description="Feature Importance Analysis of Models",
    include_package_data=True,
    install_requires=[
        "anytree",
        "cloudpickle",
        "h5py",
        "numpy>=1.19.0",
        "scikit-learn",
        "scipy",
        "sympy",
        "synmod",
        "xxhash"
    ],
    keywords="anamod",
    license="MIT",
    long_description=readme + "\n\n" + changelog,
    long_description_content_type="text/x-rst",
    name="anamod",
    packages=find_packages(),
    python_requires=">= 3.6",
    url="https://github.com/cloudbopper/anamod",
    version="0.1.1",
    zip_safe=True,
)
