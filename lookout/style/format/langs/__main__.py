import argparse
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
import logging
from pathlib import Path
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


def analyze_uast(path: str, uast: bblfsh.Node, roles: set, keywords: set, operators: set):
    KEYWORD = bblfsh.role_id("STATEMENT")
    OPERATOR = bblfsh.role_id("OPERATOR")
    file = Path(path).read_text()
    queue = [uast]
    while queue:
        node = queue.pop()
        queue.extend(node.children)
        roles.add(node.internal_type)
        if KEYWORD in node.roles:
            keywords.add(extract_node_token(file, node))
        if OPERATOR in node.roles:
            operators.add(extract_node_token(file, node))
    # Things which vmarkovtsev tried but failed:
    #
    # * use node.properties["operator"] returns instanceof, etc. for JS
    # * erase characters with assigned node.token-s, take the rest, split by whitespace,
    #   take non-alpha, put into set each char, then split alphas by those and push as keywords -
    #   yields much garbage for incorrectly parsed files.


def generate_files(outdir: str, roles: set, keywords: set, operators: set):
    env = dict(trim_blocks=True, lstrip_blocks=True)
    base = Path(__file__).parent
    outdir = Path(outdir)
    (outdir / "roles.py").write_text(
        Template((base / "roles.py.jinja2").read_text(), **env).render(roles=roles))
    (outdir / "tokens.py").write_text(
        Template((base / "tokens.py.jinja2").read_text(), **env).render(
            keywords=keywords, operators=operators))
    (outdir / "__init__.py").touch()


def main():
    args = parse_args()
    slogging.setup("INFO", False)
    clients = threading.local()
    pool = ThreadPoolExecutor(max_workers=args.threads)
    log = logging.getLogger("main")
    log.info("Will parse %d files", len(args.input))
    roles = set()
    keywords = set()
    operators = set()
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
            analyze_uast(path, response.uast, roles, keywords, operators)
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
    keywords.discard("")
    operators.discard("")
    log.info("Internal roles: %d", len(roles))
    log.info("Keywords: %d", len(keywords))
    log.info("Operators: %d", len(operators))
    generate_files(args.output, roles, keywords, operators)


if __name__ == "__main__":
    sys.exit(main())
