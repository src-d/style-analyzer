from distutils.version import LooseVersion
from importlib.machinery import SourceFileLoader
import os

import pip
from setuptools import find_packages, setup

if LooseVersion(pip.__version__) < LooseVersion("18.1"):
    raise EnvironmentError("Installation of this package requires pip >= 18.1")

lookout_style = SourceFileLoader("lookout", "./lookout/style/__init__.py").load_module()

with open(os.path.join(os.path.dirname(__file__), "README.md"), encoding="utf-8") as f:
    long_description = f.read()

tests_require = ["docker>=3.4.0,<4.0"]
tf_requires = ["tensorflow>=1.0,<2.0"]
tf_gpu_requires = ["tensorflow-gpu>=1.0,<2.0"]
plot_requires = ["matplotlib>=2.0,<3.0"]
web_requires = ["Flask>=1.0.0,<2.0", "Flask-Cors>=3.0.0,<4.0"]
all_cpu_requires = tests_require + tf_requires + plot_requires + web_requires
all_gpu_requires = tests_require + tf_gpu_requires + plot_requires + web_requires
exclude_packages = ("lookout.style.format.tests", "lookout.style.typos.tests") \
    if not os.getenv("LOOKOUT_STYLE_ANALYZER_SETUP_INCLUDE_TESTS", False) else ()

setup(
    name="lookout-style",
    description="Machine learning-based assisted code review - code style analyzers.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    version=lookout_style.__version__,
    license="AGPL-3.0",
    author="source{d}",
    author_email="machine-learning@sourced.tech",
    url="https://github.com/src-d/style-analyzer",
    download_url="https://github.com/src-d/style-analyzer",
    packages=find_packages(exclude=exclude_packages),
    namespace_packages=["lookout"],
    keywords=["machine learning on source code", "babelfish", "lookout"],
    install_requires=[
        "protobuf==3.7.1",
        "sourced-ml-core>=0.0.3,<0.1",
        "lookout-sdk-ml>=0.19.1,<0.20",
        "scikit-learn>=0.20,<2.0",
        "scikit-optimize>=0.5,<2.0",
        "pandas>=0.22,<2.0",
        "gensim>=3.7.3,<4.0",
        "google-compute-engine>=2.8.3,<3.0",  # for gensim
        "xgboost>=0.72,<2.0",
        "tabulate>=0.8.0,<2.0",
        "python-igraph>=0.7.0,<2.0",
        "smart-open==1.8.1",
        "joblib>=0.13.2,<1.0",
        "sortedcontainers>=2.1.0,<3.0",  # TODO(zurk): move to lookout-sdk-ml
        "spacy>=2.1.4,<3.0",
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
    package_data={"": ["../LICENSE.md", "../README.md", "../requirements.txt", "README.md"],
                  "lookout.style.format": ["templates/*.jinja2"],
                  "lookout.style.format.benchmarks": ["data/js_smoke_init.tar.xz",
                                                      "data/quality_report_repos.csv"],
                  "lookout.style.format.langs": ["*.jinja2"],
                  "lookout.style.format.tests": ["*.asdf", "*.xz"],
                  "lookout.style.format.tests.bugs.001_analyze_skips_lines":
                      ["find_chrome_base.js", "find_chrome_head.js",
                       "style.format.analyzer.FormatAnalyzer_1.asdf"],
                  "lookout.style.format.tests.bugs.002_bad_line_positions":
                      ["browser-policy-content.js"],
                  "lookout.style.format.tests.bugs.003_classify_vnodes_negative_col":
                      ["jquery.layout.js"],
                  "lookout.style.format.tests.bugs.004_generate_each_line":
                      ["jquery.layout.js"],
                  "lookout.style.typos": ["templates/*.jinja2", "*.xz"],
                  "lookout.style.typos.tests": ["*.asdf", "*.xz", "*.pickle", "*.bin"]},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: GNU Affero General Public License v3",
        "Operating System :: POSIX",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Topic :: Software Development :: Quality Assurance",
    ],
)
