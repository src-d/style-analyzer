"""Tooling to generate language specific resources."""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import logging
import multiprocessing
from pathlib import Path
import re
import sys
import threading

import bblfsh
from google.protobuf.message import DecodeError
from jinja2 import Template
from lookout.core import slogging
from lookout.core.cmdline import ArgumentDefaultsHelpFormatterNoNone
import pandas
from tqdm import tqdm

from lookout.style.common import handle_input_arg


def parse_args() -> argparse.Namespace:
    """Parse arguments into an argparse.Namespace."""
    parser = argparse.ArgumentParser(description="Generates a new language description for the "
                                                 "format analyzer based on the sample files.",
                                     formatter_class=ArgumentDefaultsHelpFormatterNoNone)
    parser.add_argument("--log-level", default="INFO", choices=logging._nameToLevel,
                        help="Logging verbosity.")
    parser.add_argument("--bblfsh", default="0.0.0.0:9432", help="Babelfish server's address.")
    parser.add_argument("-j", "--threads", type=int, default=multiprocessing.cpu_count() * 2,
                        help="Number of parsing workers.")
    parser.add_argument("-o", "--output", required=True, help="Output directory.")
    # We will never have here:
    #
    # 1. recursive directory support
    # 2. "glob"-style filtering
    # 3. language post-filtering
    #
    # Use find | xargs instead.
    parser.add_argument("input", nargs="+", help="Paths to the sample files. "
                                                 "Use - to read from stdin.")
    parser.add_argument("--parquet", action="store_true", default=False,
                        help="Set the input format to parquet, Must have columns: "
                             "'path', 'content', 'uast'")
    languages = ["java", "python", "go", "javascript", "typescript", "ruby", "bash", "php"]
    parser.add_argument("--parquet-language", choices=languages, default="",
                        help="The programming language to analyze. Requires --parquet.")
    return parser.parse_args()


def extract_node_token(file: str, node: bblfsh.Node) -> str:
    """
    Extract a token from a babelfish node.

    :param file: File from which the node was parsed.
    :param node: Node from which to extract a token.
    :return: Extracted token.
    """
    if node.token:
        return node.token

    if node.children:
        return ""
    spos = node.start_position
    epos = node.end_position
    if not spos or not epos:
        return ""
    return file[spos.offset:epos.offset].lower()


def analyze_uast(path: str, content: str, root: bblfsh.Node, internal_types: dict, roles: dict,
                 reserved: set):
    """
    Fill internal types, roles and reserved dictionaries with statistics computed from an UAST.

    :param path: Path of the analyzed file.
    :param content: Content of the analyzed file.
    :param root: UAST of the analyzed file.
    :param internal_types: Dictionary containing the internal types statistics.
    :param roles: Dictionary containing the roles statistics.
    :param reserved: Dictionary containing the reserved (or tokens) statistics.
    """
    # walk the tree: collect nodes with assigned tokens and build the parents map
    node_tokens = []
    parents = {}
    queue = [root]
    while queue:
        node = queue.pop()
        internal_types[node.internal_type] += 1
        for role in node.roles:
            roles[role] += 1
        for child in node.children:
            parents[id(child)] = node
        queue.extend(node.children)
        if node.token or node.start_position and node.end_position and not node.children:
            node_tokens.append(node)
    node_tokens.sort(key=lambda n: n.start_position.offset)
    sentinel = bblfsh.Node()
    sentinel.start_position.offset = len(content)
    sentinel.start_position.line = content.count("\n")
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
            content[node.start_position.offset:node.end_position.offset]
        if node.start_position.offset > pos:
            diff = content[pos:node.start_position.offset]
            parts = ws.split(diff)
            for part in parts:
                if len(part) >= 8:
                    log.debug("Skipping weird part in code: %s. Path: %s", diff, path)
                    continue
                for nonalpha in alpha.split(part):
                    for char in nonalpha:
                        if ccheck(char):
                            reserved.add(char)
        if node is sentinel:
            break
        pos = node.end_position.offset
        if IDENTIFIER not in node.roles:
            continue
        outer = content[node.start_position.offset:node.end_position.offset]
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


def generate_files(outdir: str, internal_types: dict, roles: dict, reserved: set) -> None:
    """
    Generate roles and tokens statistics modules.

    :param outdir: Output directory in which to write the computed statistics.
    :param internal_types: Internal types statistics dictionary.
    :param roles: Roles statistics dictionary.
    :param reserved: Reserved (or tokens) statistics dictionary.
    """
    env = dict(trim_blocks=True, lstrip_blocks=True)
    base = Path(__file__).parent
    outdir = Path(outdir)
    outdir.mkdir(exist_ok=True)
    (outdir / "roles.py").write_text(
        Template((base / "roles.py.jinja2").read_text(), **env).render(
            internal_types=internal_types, roles=roles))
    (outdir / "tokens.py").write_text(
        Template((base / "tokens.py.jinja2").read_text(), **env).render(reserved=reserved))
    (outdir / "__init__.py").touch()


def main():
    """Entry point."""
    args = parse_args()
    slogging.setup(args.log_level, False)
    clients = threading.local()
    pool = ThreadPoolExecutor(max_workers=args.threads)
    log = logging.getLogger("main")
    log.info("Will parse %d files in %d threads", len(args.input), args.threads)
    internal_types = defaultdict(int)
    roles = defaultdict(int)
    reserved = set()
    language = args.parquet_language
    inputs = list(handle_input_arg(args.input))
    progress = tqdm(total=len(inputs))
    progress_lock = threading.Lock()
    errors = False

    def analyze_code_file(path: str):
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
            content = Path(path).read_text()
            analyze_uast(path, content, response.uast, internal_types, roles, reserved)
        except:  # noqa: E722
            log.exception("Parsing %s", path)
            errors = True
        finally:
            with progress_lock:
                progress.disable = False  # this is needed, do not remove
                progress.update(1)

    def analyze_parquet_row(row: pandas.Series, filepath: str):
        nonlocal errors
        if errors:
            return
        nonlocal language
        try:
            path = "%s:%s" % (filepath, row.path)
            analyze_uast(path, row.content.decode(errors="ignore"),
                         bblfsh.Node.FromString(row.uast),
                         internal_types, roles, reserved)
        except DecodeError as e:
            log.warning(e)
        except:  # noqa: E722
            log.exception("Parsing %s", row.path)
            errors = True
        finally:
            with progress_lock:
                progress.disable = False  # this is needed, do not remove
                progress.update(1)
    try:
        if args.parquet:
            if not language:
                raise ValueError("--parquet-language must be specified with --parquet.")
            with progress:
                for filepath in inputs:
                    try:
                        data = pandas.read_parquet(filepath)
                    except:  # noqa: E722
                        log.warning("Bad parquet file %s", filepath)
                    else:
                        analyze = partial(analyze_parquet_row, filepath=filepath)
                        for _, row in data.iterrows():
                            progress.total += 1
                            pool.submit(analyze, row)
                    progress.update(1)
        else:
            with progress:
                for filepath in inputs:
                    pool.submit(analyze_code_file, filepath)
    finally:
        pool.shutdown()
    if errors:
        return 1
    reserved.discard("")
    log.info("Internal types: %d", len(internal_types))
    log.info("UAST roles: %d", len(roles))
    log.info("Reserved: %d", len(reserved))

    roles = {bblfsh.role_name(role_id): n for role_id, n in roles.items()}
    generate_files(args.output, internal_types, roles, reserved)


if __name__ == "__main__":
    sys.exit(main())
