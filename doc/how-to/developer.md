# Running style-analyzer in developer mode

The following steps are required to try the "format" analyzer.

1. Download [`lookout-sdk`](https://github.com/src-d/lookout/releases) binary.
2. Start the [Babelfish server](https://doc.bblf.sh/using-babelfish/getting-started.html) with the v1.2.0 javascript driver installed (`docker exec -it bblfshd bblfshctl driver install javascript bblfsh/javascript-driver:v1.2.0`)
3. Install the Python dependencies `pip3 install -e .`
4. Write the configuration file, e.g. `config.yml`:

```yaml
server: 0.0.0.0:9930
db: sqlite:////tmp/lookout.sqlite
fs: /tmp
```

5. Run the analyzer `python3 -m lookout run lookout.style.format -c config.yml`
6. Clone a Git repository and change directory there.
7. File a fake pull request `./lookout-sdk review`. It is possible to specify Git commit hashes, refer to the [manual](https://docs.sourced.tech/lookout/writing-an-analyzer/lookout-sdk#how-does-it-work).

Your git repository should contain a sufficient number of JavaScript files so that it is possible
to infer sane, statistically significant rules.