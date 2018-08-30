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
from typing import Iterable, Dict, Any

from bblfsh import Node

from lookout.core.analyzer import Analyzer, AnalyzerModel, ReferencePointer
from lookout.core.api.service_analyzer_pb2 import Comment
from lookout.core.api.service_data_pb2 import Change, File
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
    def analyze(self, ptr_from: ReferencePointer, ptr_to: ReferencePointer,
                data_request_stub: DataStub, changes: Iterable[Change]) -> [Comment]:
        self._log.info("analyze %s %s", ptr_from.commit, ptr_to.commit)
        comments = []
        for change in changes:
            comment = Comment()
            comment.file = change.head.path
            comment.text = "%s %d > %d" % (change.head.language,
                                           self.model.node_counts.get(change.base.path, 0),
                                           self.count_nodes(change.head.uast))
            comment.line = 0
            comment.confidence = 100
            comments.append(comment)
        return comments

    @classmethod
    @with_uasts_and_contents
    def train(cls, ptr: ReferencePointer, config: Dict[str, Any], data_request_stub: DataStub,
              files: Iterable[File]) -> AnalyzerModel:
        cls._log.info("train %s %s", ptr.url, ptr.commit)
        model = cls.construct_model(ptr)
        model.node_counts = {}
        for file in files:
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

`@with_changed_uasts_and_contents` and `@with_uasts_and_contents` populate the corresponding
objects with UASTs and file contents. If you need only the UASTs, use
`@with_changed_uasts` and `@with_uasts` respectively.

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
[2018-08-30T12:17:52.877425809+02:00]  INFO posting analysis app=lookout comments=1 head=HEAD provider= repository=file:///tmp/go-git
[2018-08-30T12:17:52.877452788+02:00]  INFO file comment app=lookout file=.travis.yml text=YAML 1 > 1
[2018-08-30T12:17:52.877494168+02:00]  INFO file comment app=lookout file=_examples/branch/main.go text=Go 163 > 163
[2018-08-30T12:17:52.877512614+02:00]  INFO file comment app=lookout file=blame.go text=Go 1322 > 1346
[2018-08-30T12:17:52.877529521+02:00]  INFO file comment app=lookout file=blame_test.go text=Go 987 > 1017
[2018-08-30T12:17:52.87754568+02:00]  INFO file comment app=lookout file=config/branch.go text=Go 0 > 313
[2018-08-30T12:17:52.877562436+02:00]  INFO file comment app=lookout file=config/branch_test.go text=Go 0 > 322
[2018-08-30T12:17:52.877579607+02:00]  INFO file comment app=lookout file=config/config.go text=Go 1488 > 1804
[2018-08-30T12:17:52.877610958+02:00]  INFO file comment app=lookout file=config/config_test.go text=Go 804 > 1096
[2018-08-30T12:17:52.877627674+02:00]  INFO file comment app=lookout file=config/modules.go text=Go 599 > 608
[2018-08-30T12:17:52.877647039+02:00]  INFO file comment app=lookout file=config/modules_test.go text=Go 367 > 426
[2018-08-30T12:17:52.877666745+02:00]  INFO file comment app=lookout file=config/refspec.go text=Go 679 > 686
[2018-08-30T12:17:52.877683412+02:00]  INFO file comment app=lookout file=config/refspec_test.go text=Go 717 > 865
[2018-08-30T12:17:52.877700949+02:00]  INFO file comment app=lookout file=example_test.go text=Go 416 > 444
[2018-08-30T12:17:52.877715779+02:00]  INFO file comment app=lookout file=options.go text=Go 1437 > 1509
[2018-08-30T12:17:52.87773132+02:00]  INFO global comment app=lookout text= 0 > 1
[2018-08-30T12:17:52.877746694+02:00]  INFO file comment app=lookout file=plumbing/format/gitignore/dir.go text=Go 265 > 613
[2018-08-30T12:17:52.877766641+02:00]  INFO file comment app=lookout file=plumbing/format/gitignore/dir_test.go text=Go 308 > 1560
[2018-08-30T12:17:52.877782725+02:00]  INFO file comment app=lookout file=plumbing/format/idxfile/decoder.go text=Go 725 > 724
[2018-08-30T12:17:52.877797847+02:00]  INFO file comment app=lookout file=plumbing/format/packfile/common.go text=Go 264 > 259
[2018-08-30T12:17:52.877812875+02:00]  INFO file comment app=lookout file=plumbing/format/packfile/delta_test.go text=Go 623 > 709
[2018-08-30T12:17:52.877830362+02:00]  INFO file comment app=lookout file=plumbing/format/packfile/diff_delta.go text=Go 943 > 947
[2018-08-30T12:17:52.877845029+02:00]  INFO file comment app=lookout file=plumbing/format/packfile/encoder_advanced_test.go text=Go 610 > 640
[2018-08-30T12:17:52.87786115+02:00]  INFO file comment app=lookout file=plumbing/format/packfile/index.go text=Go 420 > 746
[2018-08-30T12:17:52.877876907+02:00]  INFO file comment app=lookout file=plumbing/format/packfile/index_test.go text=Go 790 > 821
[2018-08-30T12:17:52.877893931+02:00]  INFO file comment app=lookout file=plumbing/format/packfile/scanner.go text=Go 2085 > 2093
[2018-08-30T12:17:52.877909046+02:00]  INFO file comment app=lookout file=plumbing/format/pktline/encoder.go text=Go 516 > 523
[2018-08-30T12:17:52.87792569+02:00]  INFO file comment app=lookout file=plumbing/format/pktline/scanner.go text=Go 568 > 568
[2018-08-30T12:17:52.877940972+02:00]  INFO file comment app=lookout file=plumbing/format/pktline/scanner_test.go text=Go 1221 > 1318
[2018-08-30T12:17:52.877957964+02:00]  INFO file comment app=lookout file=plumbing/object/blob.go text=Go 612 > 613
[2018-08-30T12:17:52.877972981+02:00]  INFO file comment app=lookout file=plumbing/object/change.go text=Go 641 > 734
[2018-08-30T12:17:52.877988784+02:00]  INFO file comment app=lookout file=plumbing/object/change_test.go text=Go 1961 > 2488
[2018-08-30T12:17:52.878003888+02:00]  INFO file comment app=lookout file=plumbing/object/commit.go text=Go 1899 > 1950
[2018-08-30T12:17:52.87802217+02:00]  INFO file comment app=lookout file=plumbing/object/commit_test.go text=Go 1783 > 2077
[2018-08-30T12:17:52.878037803+02:00]  INFO file comment app=lookout file=plumbing/object/commit_walker_bfs.go text=Go 0 > 419
[2018-08-30T12:17:52.878054739+02:00]  INFO file comment app=lookout file=plumbing/object/commit_walker_ctime.go text=Go 0 > 416
[2018-08-30T12:17:52.878069504+02:00]  INFO file comment app=lookout file=plumbing/object/commit_walker_test.go text=Go 616 > 1096
[2018-08-30T12:17:52.878085501+02:00]  INFO file comment app=lookout file=plumbing/object/difftree.go text=Go 109 > 170
[2018-08-30T12:17:52.878100405+02:00]  INFO file comment app=lookout file=plumbing/object/file.go text=Go 608 > 610
[2018-08-30T12:17:52.878117623+02:00]  INFO file comment app=lookout file=plumbing/object/patch.go text=Go 1378 > 1499
[2018-08-30T12:17:52.878132839+02:00]  INFO file comment app=lookout file=plumbing/object/tag.go text=Go 1599 > 1607
[2018-08-30T12:17:52.878151205+02:00]  INFO file comment app=lookout file=plumbing/object/tree.go text=Go 2085 > 2291
[2018-08-30T12:17:52.878166392+02:00]  INFO file comment app=lookout file=plumbing/object/tree_test.go text=Go 13488 > 13531
[2018-08-30T12:17:52.878182526+02:00]  INFO file comment app=lookout file=plumbing/protocol/packp/advrefs.go text=Go 553 > 922
[2018-08-30T12:17:52.878198767+02:00]  INFO file comment app=lookout file=plumbing/protocol/packp/advrefs_test.go text=Go 1356 > 1869
[2018-08-30T12:17:52.878214409+02:00]  INFO file comment app=lookout file=plumbing/storer/object.go text=Go 1259 > 1251
[2018-08-30T12:17:52.878230549+02:00]  INFO file comment app=lookout file=plumbing/transport/common.go text=Go 1161 > 1171
[2018-08-30T12:17:52.878246578+02:00]  INFO file comment app=lookout file=plumbing/transport/common_test.go text=Go 1342 > 1406
[2018-08-30T12:17:52.878262076+02:00]  INFO file comment app=lookout file=plumbing/transport/http/common.go text=Go 1106 > 1223
[2018-08-30T12:17:52.878278812+02:00]  INFO file comment app=lookout file=plumbing/transport/http/common_test.go text=Go 1030 > 1121
[2018-08-30T12:17:52.878295402+02:00]  INFO file comment app=lookout file=plumbing/transport/internal/common/common.go text=Go 1994 > 2009
[2018-08-30T12:17:52.878311716+02:00]  INFO file comment app=lookout file=plumbing/transport/internal/common/common_test.go text=Go 0 > 369
[2018-08-30T12:17:52.878327364+02:00]  INFO file comment app=lookout file=plumbing/transport/server/server.go text=Go 2138 > 2085
[2018-08-30T12:17:52.878343609+02:00]  INFO file comment app=lookout file=plumbing/transport/test/receive_pack.go text=Go 2438 > 2611
[2018-08-30T12:17:52.878358777+02:00]  INFO file comment app=lookout file=remote.go text=Go 4507 > 4739
[2018-08-30T12:17:52.878374509+02:00]  INFO file comment app=lookout file=remote_test.go text=Go 4519 > 4878
[2018-08-30T12:17:52.878390517+02:00]  INFO file comment app=lookout file=repository.go text=Go 5300 > 5962
[2018-08-30T12:17:52.878407408+02:00]  INFO file comment app=lookout file=repository_test.go text=Go 9445 > 11072
[2018-08-30T12:17:52.878422685+02:00]  INFO file comment app=lookout file=storage/filesystem/config.go text=Go 235 > 238
[2018-08-30T12:17:52.878439569+02:00]  INFO file comment app=lookout file=storage/filesystem/config_test.go text=Go 266 > 266
[2018-08-30T12:17:52.878455319+02:00]  INFO file comment app=lookout file=storage/filesystem/dotgit/dotgit.go text=Go 0 > 4094
[2018-08-30T12:17:52.878470885+02:00]  INFO file comment app=lookout file=storage/filesystem/dotgit/dotgit_rewrite_packed_refs.go text=Go 0 > 345
[2018-08-30T12:17:52.878486858+02:00]  INFO file comment app=lookout file=storage/filesystem/dotgit/dotgit_setref.go text=Go 0 > 155
[2018-08-30T12:17:52.878504272+02:00]  INFO file comment app=lookout file=storage/filesystem/dotgit/dotgit_setref_norwfs.go text=Go 0 > 185
[2018-08-30T12:17:52.878518505+02:00]  INFO file comment app=lookout file=storage/filesystem/dotgit/dotgit_test.go text=Go 0 > 4225
[2018-08-30T12:17:52.87853998+02:00]  INFO file comment app=lookout file=storage/filesystem/dotgit/writers.go text=Go 0 > 1413
[2018-08-30T12:17:52.878555832+02:00]  INFO file comment app=lookout file=storage/filesystem/dotgit/writers_test.go text=Go 0 > 884
[2018-08-30T12:17:52.878571671+02:00]  INFO file comment app=lookout file=storage/filesystem/index.go text=Go 186 > 189
[2018-08-30T12:17:52.878586346+02:00]  INFO global comment app=lookout text= 4082 > 1
[2018-08-30T12:17:52.87860415+02:00]  INFO global comment app=lookout text= 64 > 1
[2018-08-30T12:17:52.87861766+02:00]  INFO global comment app=lookout text= 117 > 1
[2018-08-30T12:17:52.878632606+02:00]  INFO global comment app=lookout text= 156 > 1
[2018-08-30T12:17:52.87864729+02:00]  INFO global comment app=lookout text= 154 > 1
[2018-08-30T12:17:52.878662021+02:00]  INFO global comment app=lookout text= 185 > 1
[2018-08-30T12:17:52.878675667+02:00]  INFO global comment app=lookout text= 4149 > 1
[2018-08-30T12:17:52.878690719+02:00]  INFO global comment app=lookout text= 1413 > 1
[2018-08-30T12:17:52.878705198+02:00]  INFO global comment app=lookout text= 884 > 1
[2018-08-30T12:17:52.878721451+02:00]  INFO file comment app=lookout file=storage/filesystem/module.go text=Go 72 > 72
[2018-08-30T12:17:52.878738932+02:00]  INFO file comment app=lookout file=storage/filesystem/object.go text=Go 2779 > 3037
[2018-08-30T12:17:52.878755606+02:00]  INFO file comment app=lookout file=storage/filesystem/object_test.go text=Go 817 > 817
[2018-08-30T12:17:52.878770143+02:00]  INFO file comment app=lookout file=storage/filesystem/reference.go text=Go 244 > 244
[2018-08-30T12:17:52.878786408+02:00]  INFO file comment app=lookout file=storage/filesystem/shallow.go text=Go 219 > 228
[2018-08-30T12:17:52.878801769+02:00]  INFO file comment app=lookout file=storage/filesystem/storage.go text=Go 207 > 207
[2018-08-30T12:17:52.878817991+02:00]  INFO file comment app=lookout file=submodule_test.go text=Go 1136 > 1292
[2018-08-30T12:17:52.878832824+02:00]  INFO file comment app=lookout file=utils/diff/diff.go text=Go 200 > 200
[2018-08-30T12:17:52.8788491+02:00]  INFO file comment app=lookout file=utils/merkletrie/difftree.go text=Go 931 > 1014
[2018-08-30T12:17:52.878869283+02:00]  INFO file comment app=lookout file=utils/merkletrie/difftree_test.go text=Go 1796 > 2197
[2018-08-30T12:17:52.87888543+02:00]  INFO file comment app=lookout file=worktree.go text=Go 3938 > 4080
[2018-08-30T12:17:52.878900631+02:00]  INFO file comment app=lookout file=worktree_bsd.go text=Go 0 > 128
[2018-08-30T12:17:52.878916457+02:00]  INFO global comment app=lookout text= 112 > 1
[2018-08-30T12:17:52.878933051+02:00]  INFO file comment app=lookout file=worktree_linux.go text=Go 112 > 128
[2018-08-30T12:17:52.878949529+02:00]  INFO file comment app=lookout file=worktree_status.go text=Go 3193 > 3215
[2018-08-30T12:17:52.878967755+02:00]  INFO file comment app=lookout file=worktree_test.go text=Go 11255 > 11711
[2018-08-30T12:17:52.878997694+02:00]  INFO file comment app=lookout file=worktree_windows.go text=Go 89 > 157
[2018-08-30T12:17:52.879017207+02:00]  INFO status: success app=lookout
```
