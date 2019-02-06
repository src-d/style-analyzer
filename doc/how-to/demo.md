# Running style-analyzer in developer mode

Install style-analyzer with `pip`:

```
pip3 install lookout-style
```

Create a fork of the GitHub repository you wish to run analysis against.
Let's suppose it is "node".
Get another [GitHub Personal Access Token](https://help.github.com/articles/creating-a-personal-access-token-for-the-command-line/) with "repo" permissions.

Then package and run the "format" analyzer:

```
analyzer package lookout.style.format -y -u github-user -t personal-access-token -r github-user/node -w /tmp/pkg
```

Here comes the tricky part: we need to override the version of JavaScript driver in Babelfish.

```
docker exec pkg_bblfsh_1 bblfshctl driver remove javascript
docker exec pkg_bblfsh_1 bblfshctl driver install javascript bblfsh/javascript-driver:v1.2.0
```

Now you can create a new pull request to your fork and watch the analysis run.
