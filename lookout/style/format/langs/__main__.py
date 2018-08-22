import argparse
from concurrent.futures import ThreadPoolExecutor
import logging
import multiprocessing
from pathlib import Path
import re
import sys
import threading

import bblfsh
from jinja2 import Template
from tqdm import tqdm

from lookout.core import slogging


def parse_args():
    parser = argparse.ArgumentParser(description="Generates a new language description for the "
                                                 "format analyser based on the sample files.")
    parser.add_argument("--bblfsh", default="0.0.0.0:9432", help="Babelfish server's address.")
    parser.add_argument("-j", "--threads", type=int, default=multiprocessing.cpu_count() * 2,
                        help="Number of parsing workers.")
    parser.add_argument("-o", "--output", required=True, help="Output directory.")
    # We will never have here:
    #
    # 1. recursive directory support
    # 2. "glob"-style filtering
    #
    # Use find | xargs instead.
    parser.add_argument("input", nargs="+", help="Paths to the sample files.")
    return parser.parse_args()


def extract_node_token(file: str, node: bblfsh.Node) -> str:
    if node.token:
        return node.token

    if node.children:
        return ""
    spos = node.start_position
    epos = node.end_position
    if not spos or not epos:
        return ""
    return file[spos.offset:epos.offset].lower()


def analyze_uast(path: str, root: bblfsh.Node, roles: set, reserved: set):
    contents = Path(path).read_text()

    # walk the tree: collect nodes with assigned tokens and build the parents map
    node_tokens = []
    parents = {}
    queue = [root]
    while queue:
        node = queue.pop()
        for child in node.children:
            parents[id(child)] = node
        queue.extend(node.children)
        if node.token or node.start_position and node.end_position and not node.children:
            node_tokens.append(node)
    node_tokens.sort(key=lambda n: n.start_position.offset)
    sentinel = bblfsh.Node()
    sentinel.start_position.offset = len(contents)
    sentinel.start_position.line = contents.count("\n")
    node_tokens.append(sentinel)

    # scan `node_tokens` and analyze the gaps and the token prefixes and suffixes
    pos = 0
    ws = re.compile("\s+")
    alpha = re.compile("[a-zA-Z]+")
    IDENTIFIER = bblfsh.role_id("IDENTIFIER")
    log = logging.getLogger("analyze_uast")

    def ccheck(char: str) -> bool:
        return not char.isspace() and not char.isalnum() and not ord(char) >= 128

    for node in node_tokens:
        token = node.token if node.token else \
            contents[node.start_position.offset:node.end_position.offset]
        if node.start_position.offset > pos:
            diff = contents[pos:node.start_position.offset]
            parts = ws.split(diff)
            for part in parts:
                if len(part) >= 8:
                    continue
                # for keyword in alpha.finditer(part):
                #    reserved.add(keyword.group())
                for nonalpha in alpha.split(part):
                    for char in nonalpha:
                        if ccheck(char):
                            reserved.add(char)
        if node is sentinel:
            break
        pos = node.end_position.offset
        if IDENTIFIER not in node.roles:
            continue
        outer = contents[node.start_position.offset:node.end_position.offset]
        if outer == token:
            continue
        pos = outer.find(token)
        if pos < 0:
            log.warning("skipped %s, token offset corruption \"%s\" vs. \"%s\"",
                        path, token, outer)
            break
        if pos > 0:
            for char in outer[:pos]:
                if ccheck(char):
                    reserved.add(char)
        if pos + len(token) < len(outer):
            for char in outer[pos + len(token):]:
                if ccheck(char):
                    reserved.add(char)


def generate_files(outdir: str, roles: set, reserved: set):
    env = dict(trim_blocks=True, lstrip_blocks=True)
    base = Path(__file__).parent
    outdir = Path(outdir)
    outdir.mkdir(exist_ok=True)
    (outdir / "roles.py").write_text(
        Template((base / "roles.py.jinja2").read_text(), **env).render(roles=roles))
    (outdir / "tokens.py").write_text(
        Template((base / "tokens.py.jinja2").read_text(), **env).render(reserved=reserved))
    (outdir / "__init__.py").touch()


def main():
    args = parse_args()
    slogging.setup("INFO", False)
    clients = threading.local()
    pool = ThreadPoolExecutor(max_workers=args.threads)
    log = logging.getLogger("main")
    log.info("Will parse %d files", len(args.input))
    roles = set()
    reserved = set()
    language = ""
    progress = tqdm(total=len(args.input))
    errors = False

    def analyze_file(path: str):
        nonlocal errors
        if errors:
            return
        try:
            try:
                client = clients.client
            except AttributeError:
                client = bblfsh.BblfshClient(args.bblfsh)
                clients.client = client
            response = client.parse(path)
            nonlocal language
            if not language:
                language = response.language
            elif language != response.language:
                log.warning("dropped %s - language mismatch %s != %s",
                            path, language, response.language)
                return
            analyze_uast(path, response.uast, roles, reserved)
            progress.update(1)
        except:  # noqa: E722
            log.exception("Parsing %s", path)
            errors = True

    with progress:
        for file in args.input:
            pool.submit(analyze_file, file)
        pool.shutdown()
    if errors:
        return 1
    reserved.discard("")
    log.info("Internal roles: %d", len(roles))
    log.info("Reserved: %d", len(reserved))
    generate_files(args.output, roles, reserved)


if __name__ == "__main__":
    sys.exit(main())
