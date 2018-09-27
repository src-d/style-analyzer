import argparse
import importlib
import json
import logging
import pkgutil
import sys  # noqa: F401
from unittest.mock import patch

import configargparse
import humanfriendly

import lookout
from lookout.core import slogging
from lookout.core.data_requests import DataService
from lookout.core.event_listener import EventListener
from lookout.core.manager import AnalyzerManager
from lookout.core.sqla_model_repository import SQLAlchemyModelRepository


class ArgumentDefaultsHelpFormatterNoNone(argparse.ArgumentDefaultsHelpFormatter):
    """
    Pretty formatter of help message for arguments.
    It adds default value to the end if it is not None.
    """
    def _get_help_string(self, action):
        if action.default is None:
            return action.help
        return super()._get_help_string(action)


def list_analyzers(args):
    """
    Prints the list of the analyzers inside the `lookout` package.

    :param args: Not used - parsed command line arguments.
    :return: None
    """
    first = True
    queue = [tuple(c) + ("lookout.",) for c in pkgutil.iter_modules(lookout.__path__)]
    while queue:
        importer, name, ispkg, prefix = queue.pop(0)

        if not ispkg or name == "core":
            continue

        m = importer.find_module(name).load_module(name)
        if getattr(m, "__meta__", False):
            queue.extend(tuple(c) + (prefix + name + ".",)
                         for c in pkgutil.iter_modules(m.__path__))
            continue

        if first:
            first = False
        else:
            print()
        print(prefix + name)
        cls = m.analyzer_class
        print("\t" + cls.version)
        print("\t" + cls.description)


def run_analyzers(args):
    """
    Launches the service with the specified analyzers. Blocks until a KeyboardInterrupt.

    :param args: Parsed command line arguments.
    :return: None
    """
    slogging.setup(args.log_level, args.log_structured, args.log_config_path)
    log = logging.getLogger("run")
    model_repository = create_model_repo_from_args(args)
    log.info("Created %s", model_repository)
    if args.request_server == "auto":
        data_request_address = "%s:10301" % args.server.split(":")[0]
    else:
        data_request_address = args.request_server
    data_service = DataService(data_request_address)
    log.info("Created %s", data_service)
    manager = AnalyzerManager(
        analyzers=[importlib.import_module(a).analyzer_class for a in args.analyzer],
        model_repository=model_repository,
        data_service=data_service,
    )
    log.info("Created %s", manager)
    listener = EventListener(address=args.server, handlers=manager, n_workers=args.workers)
    log.info("Created %s", listener)
    listener.start()
    log.info("Listening %s", args.server)
    listener.block()
    model_repository.shutdown()
    data_service.shutdown()


def init_repo(args):
    """
    Initializes the model repository.

    :param args: Parsed command line arguments.
    :return: None
    """
    slogging.setup(args.log_level, False, args.log_config_path)
    repo = create_model_repo_from_args(args)
    repo.init()


def create_model_repo_from_args(args) -> SQLAlchemyModelRepository:
    return SQLAlchemyModelRepository(
        db_endpoint=args.db, fs_root=args.fs,
        max_cache_mem=humanfriendly.parse_size(args.cache_size),
        ttl=int(humanfriendly.parse_timespan(args.cache_ttl)),
        engine_kwargs=args.db_kwargs)


def add_model_repository_args(parser):
    parser.add("-d", "--db", required=True, help="Model repository database address.")
    parser.add("-f", "--fs", required=True, help="Model repository file system root.")
    parser.add("--cache-size", default="1G",
               help="Model repository cache size - accepts human-readable values like 200M, 2G.")
    parser.add("--cache-ttl", default="6h",
               help="Model repository cache time-to-live (TTL) - accepts human-readable "
                    "values like 30min, 4h, 1d.")
    parser.add("--db-kwargs", type=json.loads, default={},
               help="Additional keyword arguments to SQLAlchemy database engine.")


def tool(args) -> None:
    """
    Invoke the tooling of an analyzer.

    :param args: Parsed command line arguments.
    :return: None
    """
    with patch("sys.argv", [args.analyzer] + args.args):
        importlib.import_module(args.analyzer).run_cmdline_tool()


def add_logging_args(parser):
    parser.add("--log-level", default="INFO", choices=logging._nameToLevel,
               help="Logging verbosity.")
    parser.add("--log-config-path",
               help="Path to the file which sets individual log levels of domains.")


def create_parser():
    parser = configargparse.ArgParser(default_config_files=[
        "/etc/lookout/analyzer.conf", "~/.config/lookout/analyzer.conf"],
        formatter_class=ArgumentDefaultsHelpFormatterNoNone,
        auto_env_var_prefix="lookout_")
    subparsers = parser.add_subparsers(help="Commands", dest="command")

    def add_parser(name, help):
        return subparsers.add_parser(
            name, help=help, formatter_class=ArgumentDefaultsHelpFormatterNoNone)

    list_parser = add_parser("list", "Print globally available analyzers.")
    list_parser.set_defaults(handler=list_analyzers)

    run_parser = add_parser(
        "run", "Launch a new service with the specified (one or more) analyzers.")
    run_parser.set_defaults(handler=run_analyzers)
    add_logging_args(run_parser)
    run_parser.add("--log-structured", action="store_true",
                   help="Enable structured logging (compatible with k8s).")
    run_parser.add("-c", "--config", is_config_file=True,
                   help="Path to the configuration file with option defaults.")
    run_parser.add("analyzer", nargs="+", help="Fully qualified package name with an analyzer.")
    run_parser.add("-s", "--server", required=True,
                   help="Lookout server address, e.g. localhost:1234.")
    run_parser.add("-w", "--workers", type=int, default=1,
                   help="Number of threads which process Lookout events.")
    add_model_repository_args(run_parser)
    run_parser.add_argument("--request-server", default="auto",
                            help="Address of the data retrieval service. \"same\" means --server.")

    init_parser = add_parser("init", "Initialize the model repository.")
    init_parser.set_defaults(handler=init_repo)
    add_model_repository_args(init_parser)
    add_logging_args(init_parser)

    tool_parser = add_parser("tool", "Invoke the tooling of a given analyzer.")
    tool_parser.set_defaults(handler=tool)
    tool_parser.add("analyzer", help="Fully qualified package name with an analyzer.")
    tool_parser.add("args", nargs=argparse.REMAINDER)
    return parser
