# Getting started with Lookout Python SDK

This is the guide on how to develop a Lookout Analyzer in Python.

## Environment

* Linux or macOS
* 3.5 <= Python <= 3.7
* Babelfish
* Lookout Server

Install the SDK:

```
pip install lookout-style
```

[Install and run bblfshd.](https://doc.bblf.sh/using-babelfish/getting-started.html)

Download the [`lookout`](https://github.com/src-d/lookout/releases) binary. The version must
match [the submodule revision](../server).

Create the configuration `config.yml`:

```yaml
server: 0.0.0.0:2000
db: sqlite:////tmp/lookout.sqlite
fs: /tmp
```

Clone a sample project to test on:

```
git clone https://github.com/src-d/go-git /tmp/go-git
```

## Sample code

Create `my_analyzer.py`:

```python
import logging

from bblfsh import Node

from lookout.core.analyzer import Analyzer, AnalyzerModel
from lookout.core.api.service_analyzer_pb2 import Comment
from lookout.core.api.service_data_pb2_grpc import DataStub
from lookout.core.data_requests import with_changed_uasts_and_contents, with_uasts_and_contents


class MyModel(AnalyzerModel):
    NAME = "my-model"
    VENDOR = "source{d}"

    def _load_tree(self, tree: dict) -> None:
        self.node_counts = tree["node_counts"]

    def _generate_tree(self) -> dict:
        return {"node_counts": self.node_counts}


class MyAnalyzer(Analyzer):
    model_type = MyModel
    version = "1"
    description = "Reports the changes in UAST node counts."
    _log = logging.getLogger("MyAnalyzer")
    
    @with_changed_uasts_and_contents
    def analyze(self, commit_from: str, commit_to: str, data_request_stub: DataStub,
                **data) -> [Comment]:
        self._log.info("analyze %s %s", commit_from, commit_to)
        comments = []
        for change in data["changes"]:
            comment = Comment()
            comment.file = change.head.path
            comment.text = "%s %d > %d" % (change.head.language,
                                           self.model.node_counts.get(change.base.path, 0),
                                           self.count_nodes(change.head.uast))
            comment.line = 1
            comment.confidence = 100
            comments.append(comment)
        return comments

    @classmethod
    @with_uasts_and_contents
    def train(cls, url: str, commit: str, config: dict, data_request_stub: DataStub,
              **data) -> AnalyzerModel:
        cls._log.info("train %s %s", url, commit)
        model = MyModel().construct(cls, url, commit)
        model.node_counts = {}
        for file in data["files"]:
            model.node_counts[file.path] = cls.count_nodes(file.uast)
        return model

    @staticmethod
    def count_nodes(uast: Node):
        stack = [uast]
        count = 0
        while stack:
            node = stack.pop()
            count += 1
            stack.extend(node.children)
        return count
        
analyzer_class = MyAnalyzer
```

## Running

Launch the analyzer:

```
analyzer run my_analyzer -c config.yml
```

Train the model:

```
lookout push ipv4://localhost:2000 --git-dir /tmp/go-git \
    --from defd0b861ca79845c8f06f7c826c769012404bbd \
    --to 4397264e391b45a8eac147cc7373189d55c640cc
```

You should have `/tmp/tmp/go-git/MyAnalyzer_1.asdf`.

Run the model:

```
lookout review ipv4://localhost:2000 --git-dir /tmp/go-git \
    --from 4397264e391b45a8eac147cc7373189d55c640cc \
    --to 9f00789688d26191a987fdec8bc2678362ec4453
```

You should see the comments in the log:

```
[2018-08-23T18:07:20.633172637+02:00]  INFO line comment app=lookout file=.travis.yml line=1 text=YAML 1 > 1
[2018-08-23T18:07:20.63323373+02:00]  INFO line comment app=lookout file=_examples/branch/main.go line=1 text=Go 163 > 163
[2018-08-23T18:07:20.633251954+02:00]  INFO line comment app=lookout file=blame.go line=1 text=Go 1322 > 1346
[2018-08-23T18:07:20.633268069+02:00]  INFO line comment app=lookout file=blame_test.go line=1 text=Go 987 > 1017
[2018-08-23T18:07:20.633282103+02:00]  INFO line comment app=lookout file=config/branch.go line=1 text=Go 0 > 313
[2018-08-23T18:07:20.633298621+02:00]  INFO line comment app=lookout file=config/branch_test.go line=1 text=Go 0 > 322
[2018-08-23T18:07:20.63331431+02:00]  INFO line comment app=lookout file=config/config.go line=1 text=Go 1488 > 1804
[2018-08-23T18:07:20.633329044+02:00]  INFO line comment app=lookout file=config/config_test.go line=1 text=Go 804 > 1096
[2018-08-23T18:07:20.63334381+02:00]  INFO line comment app=lookout file=config/modules.go line=1 text=Go 599 > 608
[2018-08-23T18:07:20.63335966+02:00]  INFO line comment app=lookout file=config/modules_test.go line=1 text=Go 367 > 426
[2018-08-23T18:07:20.633373079+02:00]  INFO line comment app=lookout file=config/refspec.go line=1 text=Go 679 > 686
[2018-08-23T18:07:20.633389259+02:00]  INFO line comment app=lookout file=config/refspec_test.go line=1 text=Go 717 > 865
[2018-08-23T18:07:20.633403361+02:00]  INFO line comment app=lookout file=example_test.go line=1 text=Go 416 > 444
[2018-08-23T18:07:20.633420242+02:00]  INFO line comment app=lookout file=options.go line=1 text=Go 1437 > 1509
[2018-08-23T18:07:20.633449869+02:00]  INFO global comment app=lookout text= 0 > 1
[2018-08-23T18:07:20.633464179+02:00]  INFO line comment app=lookout file=plumbing/format/gitignore/dir.go line=1 text=Go 265 > 613
[2018-08-23T18:07:20.633479206+02:00]  INFO line comment app=lookout file=plumbing/format/gitignore/dir_test.go line=1 text=Go 308 > 1560
[2018-08-23T18:07:20.633495512+02:00]  INFO line comment app=lookout file=plumbing/format/idxfile/decoder.go line=1 text=Go 725 > 724
[2018-08-23T18:07:20.633510201+02:00]  INFO line comment app=lookout file=plumbing/format/packfile/common.go line=1 text=Go 264 > 259
[2018-08-23T18:07:20.633524714+02:00]  INFO line comment app=lookout file=plumbing/format/packfile/delta_test.go line=1 text=Go 623 > 709
[2018-08-23T18:07:20.633539347+02:00]  INFO line comment app=lookout file=plumbing/format/packfile/diff_delta.go line=1 text=Go 943 > 947
[2018-08-23T18:07:20.633555123+02:00]  INFO line comment app=lookout file=plumbing/format/packfile/encoder_advanced_test.go line=1 text=Go 610 > 640
[2018-08-23T18:07:20.633570142+02:00]  INFO line comment app=lookout file=plumbing/format/packfile/index.go line=1 text=Go 420 > 746
[2018-08-23T18:07:20.633586756+02:00]  INFO line comment app=lookout file=plumbing/format/packfile/index_test.go line=1 text=Go 790 > 821
[2018-08-23T18:07:20.633601678+02:00]  INFO line comment app=lookout file=plumbing/format/packfile/scanner.go line=1 text=Go 2085 > 2093
[2018-08-23T18:07:20.633616713+02:00]  INFO line comment app=lookout file=plumbing/format/pktline/encoder.go line=1 text=Go 516 > 523
[2018-08-23T18:07:20.633633318+02:00]  INFO line comment app=lookout file=plumbing/format/pktline/scanner.go line=1 text=Go 568 > 568
[2018-08-23T18:07:20.633648859+02:00]  INFO line comment app=lookout file=plumbing/format/pktline/scanner_test.go line=1 text=Go 1221 > 1318
[2018-08-23T18:07:20.63366347+02:00]  INFO line comment app=lookout file=plumbing/object/blob.go line=1 text=Go 612 > 613
[2018-08-23T18:07:20.633680165+02:00]  INFO line comment app=lookout file=plumbing/object/change.go line=1 text=Go 641 > 734
[2018-08-23T18:07:20.633695275+02:00]  INFO line comment app=lookout file=plumbing/object/change_test.go line=1 text=Go 1961 > 2488
[2018-08-23T18:07:20.633709674+02:00]  INFO line comment app=lookout file=plumbing/object/commit.go line=1 text=Go 1899 > 1950
[2018-08-23T18:07:20.633724015+02:00]  INFO line comment app=lookout file=plumbing/object/commit_test.go line=1 text=Go 1783 > 2077
[2018-08-23T18:07:20.633738427+02:00]  INFO line comment app=lookout file=plumbing/object/commit_walker_bfs.go line=1 text=Go 0 > 419
[2018-08-23T18:07:20.633754763+02:00]  INFO line comment app=lookout file=plumbing/object/commit_walker_ctime.go line=1 text=Go 0 > 416
[2018-08-23T18:07:20.633771727+02:00]  INFO line comment app=lookout file=plumbing/object/commit_walker_test.go line=1 text=Go 616 > 1096
[2018-08-23T18:07:20.633785512+02:00]  INFO line comment app=lookout file=plumbing/object/difftree.go line=1 text=Go 109 > 170
[2018-08-23T18:07:20.633800497+02:00]  INFO line comment app=lookout file=plumbing/object/file.go line=1 text=Go 608 > 610
[2018-08-23T18:07:20.633819542+02:00]  INFO line comment app=lookout file=plumbing/object/patch.go line=1 text=Go 1378 > 1499
[2018-08-23T18:07:20.633836051+02:00]  INFO line comment app=lookout file=plumbing/object/tag.go line=1 text=Go 1599 > 1607
[2018-08-23T18:07:20.633850879+02:00]  INFO line comment app=lookout file=plumbing/object/tree.go line=1 text=Go 2085 > 2291
[2018-08-23T18:07:20.633866677+02:00]  INFO line comment app=lookout file=plumbing/object/tree_test.go line=1 text=Go 13488 > 13531
[2018-08-23T18:07:20.633883849+02:00]  INFO line comment app=lookout file=plumbing/protocol/packp/advrefs.go line=1 text=Go 553 > 922
[2018-08-23T18:07:20.633899603+02:00]  INFO line comment app=lookout file=plumbing/protocol/packp/advrefs_test.go line=1 text=Go 1356 > 1869
[2018-08-23T18:07:20.633913203+02:00]  INFO line comment app=lookout file=plumbing/storer/object.go line=1 text=Go 1259 > 1251
[2018-08-23T18:07:20.633929226+02:00]  INFO line comment app=lookout file=plumbing/transport/common.go line=1 text=Go 1161 > 1171
[2018-08-23T18:07:20.633943942+02:00]  INFO line comment app=lookout file=plumbing/transport/common_test.go line=1 text=Go 1342 > 1406
[2018-08-23T18:07:20.633961171+02:00]  INFO line comment app=lookout file=plumbing/transport/http/common.go line=1 text=Go 1106 > 1223
[2018-08-23T18:07:20.633977114+02:00]  INFO line comment app=lookout file=plumbing/transport/http/common_test.go line=1 text=Go 1030 > 1121
[2018-08-23T18:07:20.633993826+02:00]  INFO line comment app=lookout file=plumbing/transport/internal/common/common.go line=1 text=Go 1994 > 2009
[2018-08-23T18:07:20.634010212+02:00]  INFO line comment app=lookout file=plumbing/transport/internal/common/common_test.go line=1 text=Go 0 > 369
[2018-08-23T18:07:20.634027963+02:00]  INFO line comment app=lookout file=plumbing/transport/server/server.go line=1 text=Go 2138 > 2085
[2018-08-23T18:07:20.634042978+02:00]  INFO line comment app=lookout file=plumbing/transport/test/receive_pack.go line=1 text=Go 2438 > 2611
[2018-08-23T18:07:20.634059168+02:00]  INFO line comment app=lookout file=remote.go line=1 text=Go 4507 > 4739
[2018-08-23T18:07:20.634073989+02:00]  INFO line comment app=lookout file=remote_test.go line=1 text=Go 4519 > 4878
[2018-08-23T18:07:20.634094376+02:00]  INFO line comment app=lookout file=repository.go line=1 text=Go 5300 > 5962
[2018-08-23T18:07:20.634112927+02:00]  INFO line comment app=lookout file=repository_test.go line=1 text=Go 9445 > 11072
[2018-08-23T18:07:20.634128353+02:00]  INFO line comment app=lookout file=storage/filesystem/config.go line=1 text=Go 235 > 238
[2018-08-23T18:07:20.634143967+02:00]  INFO line comment app=lookout file=storage/filesystem/config_test.go line=1 text=Go 266 > 266
[2018-08-23T18:07:20.6341591+02:00]  INFO line comment app=lookout file=storage/filesystem/dotgit/dotgit.go line=1 text=Go 0 > 4094
[2018-08-23T18:07:20.634175398+02:00]  INFO line comment app=lookout file=storage/filesystem/dotgit/dotgit_rewrite_packed_refs.go line=1 text=Go 0 > 345
[2018-08-23T18:07:20.634191619+02:00]  INFO line comment app=lookout file=storage/filesystem/dotgit/dotgit_setref.go line=1 text=Go 0 > 155
[2018-08-23T18:07:20.634207242+02:00]  INFO line comment app=lookout file=storage/filesystem/dotgit/dotgit_setref_norwfs.go line=1 text=Go 0 > 185
[2018-08-23T18:07:20.634221744+02:00]  INFO line comment app=lookout file=storage/filesystem/dotgit/dotgit_test.go line=1 text=Go 0 > 4225
[2018-08-23T18:07:20.634239855+02:00]  INFO line comment app=lookout file=storage/filesystem/dotgit/writers.go line=1 text=Go 0 > 1413
[2018-08-23T18:07:20.634257789+02:00]  INFO line comment app=lookout file=storage/filesystem/dotgit/writers_test.go line=1 text=Go 0 > 884
[2018-08-23T18:07:20.634274706+02:00]  INFO line comment app=lookout file=storage/filesystem/index.go line=1 text=Go 186 > 189
[2018-08-23T18:07:20.634289607+02:00]  INFO global comment app=lookout text= 4082 > 1
[2018-08-23T18:07:20.63430321+02:00]  INFO global comment app=lookout text= 64 > 1
[2018-08-23T18:07:20.63431716+02:00]  INFO global comment app=lookout text= 117 > 1
[2018-08-23T18:07:20.634330964+02:00]  INFO global comment app=lookout text= 156 > 1
[2018-08-23T18:07:20.634343172+02:00]  INFO global comment app=lookout text= 154 > 1
[2018-08-23T18:07:20.634354942+02:00]  INFO global comment app=lookout text= 185 > 1
[2018-08-23T18:07:20.63436786+02:00]  INFO global comment app=lookout text= 4149 > 1
[2018-08-23T18:07:20.634380812+02:00]  INFO global comment app=lookout text= 1413 > 1
[2018-08-23T18:07:20.634395626+02:00]  INFO global comment app=lookout text= 884 > 1
[2018-08-23T18:07:20.634408848+02:00]  INFO line comment app=lookout file=storage/filesystem/module.go line=1 text=Go 72 > 72
[2018-08-23T18:07:20.634425607+02:00]  INFO line comment app=lookout file=storage/filesystem/object.go line=1 text=Go 2779 > 3037
[2018-08-23T18:07:20.634439572+02:00]  INFO line comment app=lookout file=storage/filesystem/object_test.go line=1 text=Go 817 > 817
[2018-08-23T18:07:20.634455417+02:00]  INFO line comment app=lookout file=storage/filesystem/reference.go line=1 text=Go 244 > 244
[2018-08-23T18:07:20.634472189+02:00]  INFO line comment app=lookout file=storage/filesystem/shallow.go line=1 text=Go 219 > 228
[2018-08-23T18:07:20.634487904+02:00]  INFO line comment app=lookout file=storage/filesystem/storage.go line=1 text=Go 207 > 207
[2018-08-23T18:07:20.634502536+02:00]  INFO line comment app=lookout file=submodule_test.go line=1 text=Go 1136 > 1292
[2018-08-23T18:07:20.63451906+02:00]  INFO line comment app=lookout file=utils/diff/diff.go line=1 text=Go 200 > 200
[2018-08-23T18:07:20.634532945+02:00]  INFO line comment app=lookout file=utils/merkletrie/difftree.go line=1 text=Go 931 > 1014
[2018-08-23T18:07:20.634549704+02:00]  INFO line comment app=lookout file=utils/merkletrie/difftree_test.go line=1 text=Go 1796 > 2197
[2018-08-23T18:07:20.634565074+02:00]  INFO line comment app=lookout file=worktree.go line=1 text=Go 3938 > 4080
[2018-08-23T18:07:20.634580781+02:00]  INFO line comment app=lookout file=worktree_bsd.go line=1 text=Go 0 > 128
[2018-08-23T18:07:20.634594177+02:00]  INFO global comment app=lookout text= 112 > 1
[2018-08-23T18:07:20.634610128+02:00]  INFO line comment app=lookout file=worktree_linux.go line=1 text=Go 112 > 128
[2018-08-23T18:07:20.63462568+02:00]  INFO line comment app=lookout file=worktree_status.go line=1 text=Go 3193 > 3215
[2018-08-23T18:07:20.634641147+02:00]  INFO line comment app=lookout file=worktree_test.go line=1 text=Go 11255 > 11711
[2018-08-23T18:07:20.634657451+02:00]  INFO line comment app=lookout file=worktree_windows.go line=1 text=Go 89 > 157
```