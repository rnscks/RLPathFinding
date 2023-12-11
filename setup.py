from setuptools import setup, find_packages

setup(
    name="path_finding_map",
    version="0.1.0",
    author="rnscks",
    author_email="mplngx2@gmail.com",
    description="path finding map generator and convertor",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/my_package",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
