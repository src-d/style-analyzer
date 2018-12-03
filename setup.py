from importlib.machinery import SourceFileLoader
import os

from setuptools import find_packages, setup

lookout_style = SourceFileLoader("lookout", "./lookout/style/__init__.py").load_module()

with open(os.path.join(os.path.dirname(__file__), "README.md")) as f:
    long_description = f.read()

tests_require = ["docker>=3.4.0,<4.0"]
tf_requires = ["tensorflow>=1.0,<2.0"]
tf_gpu_requires = ["tensorflow-gpu>=1.0,<2.0"]
plot_requires = ["matplotlib>=2.0,<3.0"]
web_requires = ["Flask>=1.0.0,<2.0", "Flask-Cors>=3.0.0,<4.0"]
all_cpu_requires = tests_require + tf_requires + plot_requires + web_requires
all_gpu_requires = tests_require + tf_gpu_requires + plot_requires + web_requires

setup(
    name="lookout-style",
    description="Machine learning-based assisted code review - code style analyzers.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    version=".".join(map(str, lookout_style.__version__)),
    license="AGPL-3.0",
    author="source{d}",
    author_email="machine-learning@sourced.tech",
    url="https://github.com/src-d/style-analyzer",
    download_url="https://github.com/src-d/style-analyzer",
    packages=find_packages(exclude=("lookout.style.format.tests",)),
    namespace_packages=["lookout"],
    keywords=["machine learning on source code", "babelfish", "lookout"],
    install_requires=[
        "sourced-ml>=0.7.0,<0.8",
        "lookout-sdk-ml>=0.3,<0.4",
        "scikit-learn>=0.20,<2.0",
        "scikit-optimize>=0.5,<2.0",
        "pandas>=0.22,<2.0",
        "gensim>=3.5.0,<4.0",
        "google-compute-engine>=2.8.3,<3.0",  # for gensim
        "xgboost>=0.72,<2.0",
    ],
    extras_require={
        "tf": tf_requires,
        "tf_gpu": tf_gpu_requires,
        "plot": plot_requires,
        "test": tests_require,
        "web": web_requires,
        "all_gpu": all_gpu_requires,
        "all_cpu": all_cpu_requires,
    },
    tests_require=tests_require,
    package_data={"": ["LICENSE.md", "README.md"], },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: GNU Affero General Public License v3",
        "Operating System :: POSIX",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Topic :: Software Development :: Quality Assurance"
    ]
)
