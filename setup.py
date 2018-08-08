from importlib.machinery import SourceFileLoader
from setuptools import setup, find_packages
import sys

lookout = SourceFileLoader("lookout", "./lookout/__init__.py").load_module()

if sys.version_info < (3, 5, 0):
    typing = ["typing"]
else:
    typing = []

setup(
    name="lookout-style",
    description="Machine learning-based assisted code review - code style analyzers.",
    version=".".join(map(str, lookout.__version__)),
    license="AGPL-3.0",
    author="source{d}",
    author_email="machine-learning@sourced.tech",
    url="https://github.com/src-d/style-analyzer",
    download_url="https://github.com/src-d/style-analyzer",
    packages=find_packages(exclude=("lookout.style.format.tests",)),
    namespace_packages=["lookout"],
    entry_points={
        "console_scripts": ["analyzer=lookout.__main__:main"],
    },
    keywords=["machine learning on source code", "babelfish"],
    install_requires=["sourced-ml>=0.5.1,<0.6",
                      "xxhash>=0.5.0,<2.0",
                      "stringcase>=1.2.0,<2.0",
                      "sqlalchemy>=1.0.0,<2.0",
                      "pympler>=0.5,<2.0",
                      "cachetools>=2.0,<3.0",
                      "configargparse>=0.13,<2.0",
                      "humanfriendly>=4.0,<5.0",
                      "psycopg2-binary>=2.7,<3.0",
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
