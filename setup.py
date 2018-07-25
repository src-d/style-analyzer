from importlib.machinery import SourceFileLoader
from setuptools import setup, find_packages
import sys

lookout_style = SourceFileLoader("lookout.style", "./lookout/style/__init__.py").load_module()

if sys.version_info < (3, 5, 0):
    typing = ["typing"]
else:
    typing = []

setup(
    name="lookout-style",
    description="Machine learning-based assisted code review - code style analyzers.",
    version=".".join(map(str, lookout_style.__version__)),
    license="AGPL-3.0",
    author="source{d}",
    author_email="machine-learning@sourced.tech",
    url="https://github.com/src-d/style-analyzer",
    download_url="https://github.com/src-d/style-analyzer",
    packages=find_packages(exclude=("lookout.style.format.tests",)),
    namespace_packages=["lookout"],
    entry_points={
        "console_scripts": ["lookout-style=lookout.style.__main__:main"],
    },
    keywords=["machine learning on source code", "babelfish"],
    install_requires=["sourced-ml>=0.5.1,<0.6",
                      ] + typing,
    extras_require={
        "tf": ["tensorflow>=1.0,<2.0"],
        "tf_gpu": ["tensorflow-gpu>=1.0,<2.0"],
    },
    tests_require=["docker>=3.4.0,<4.0"],
    package_data={"": ["LICENSE.md", "README.md"], },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: GNU Affero General Public License v3",
        "Operating System :: POSIX",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Topic :: Software Development :: Quality Assurance"
    ]
)
