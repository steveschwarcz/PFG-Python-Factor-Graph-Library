import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pfg",
    version="1.0.1",
    author="Steven Schwarcz",
    author_email="sasz11@gmail.com",
    description="A lightweight library for building Factor Graphs and performing inference using the loopy belief propagation algorithm.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/steveschwarcz/PFG-Python-Factor-Graph-Library",
    packages=setuptools.find_packages(),
    install_requires=[
                      'numpy',
                      ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)