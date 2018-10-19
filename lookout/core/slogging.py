import codecs
import datetime
import io
import json
import logging
import re
import sys
import threading
import traceback
from typing import Dict, Sequence, Union

import numpy
import xxhash
import yaml


logs_are_structured = False


def get_timezone():
    dt = datetime.datetime.now(datetime.timezone.utc).astimezone()
    tzstr = dt.strftime("%z")
    tzstr = tzstr[:-2] + ":" + tzstr[-2:]
    return dt.tzinfo, tzstr


timezone, tzstr = get_timezone()


def format_datetime(dt: datetime.datetime):
    return dt.strftime("%Y-%m-%dT%k:%M:%S.%f000") + tzstr


def reduce_thread_id(thread_id: int) -> str:
    return xxhash.xxh32(thread_id.to_bytes(8, "little")).hexdigest()[:4]


class NumpyLogRecord(logging.LogRecord):
    """
    LogRecord with the special handling of numpy arrays which shortens the long ones.
    """

    @staticmethod
    def array2string(arr: numpy.ndarray) -> str:
        shape = str(arr.shape)[1:-1]
        if shape.endswith(","):
            shape = shape[:-1]
        return numpy.array2string(arr, threshold=11) + "%s[%s]" % (arr.dtype, shape)

    def getMessage(self):
        """
        Return the message for this LogRecord.

        Return the message for this LogRecord after merging any user-supplied
        arguments with the message.
        """
        if isinstance(self.msg, numpy.ndarray):
            msg = self.array2string(self.msg)
        else:
            msg = str(self.msg)
        if self.args:
            a2s = self.array2string
            if isinstance(self.args, Dict):
                args = {k: (a2s(v) if isinstance(v, numpy.ndarray) else v)
                        for (k, v) in self.args.items()}
            elif isinstance(self.args, Sequence):
                args = tuple((a2s(a) if isinstance(a, numpy.ndarray) else a)
                             for a in self.args)
            else:
                raise TypeError("Unexpected input '%s' with type '%s'" % (self.args,
                                                                          type(self.args)))
            msg = msg % args
        return msg


class AwesomeFormatter(logging.Formatter):
    """
    logging.Formatter which adds colors to messages and shortens thread ids.
    """
    GREEN_MARKERS = [" ok", "ok:", "finished", "complete", "ready",
                     "done", "running", "success", "saved"]
    GREEN_RE = re.compile("|".join(GREEN_MARKERS))

    def formatMessage(self, record: logging.LogRecord) -> str:
        level_color = "0"
        text_color = "0"
        fmt = ""
        if record.levelno <= logging.DEBUG:
            fmt = "\033[0;37m" + logging.BASIC_FORMAT + "\033[0m"
        elif record.levelno <= logging.INFO:
            level_color = "1;36"
            lmsg = record.message.lower()
            if self.GREEN_RE.search(lmsg):
                text_color = "1;32"
        elif record.levelno <= logging.WARNING:
            level_color = "1;33"
        elif record.levelno <= logging.CRITICAL:
            level_color = "1;31"
        if not fmt:
            fmt = "\033[" + level_color + \
                  "m%(levelname)s\033[0m:%(rthread)s:%(name)s:\033[" + text_color + \
                  "m%(message)s\033[0m"
        record.rthread = reduce_thread_id(record.thread)
        return fmt % record.__dict__


class StructuredHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)
        self.local = threading.local()

    def emit(self, record: logging.LogRecord):
        created = datetime.datetime.fromtimestamp(record.created, timezone)
        obj = {
            "level": record.levelname.lower(),
            "msg": record.msg % record.args,
            "source": "%s:%d" % (record.filename, record.lineno),
            "time": format_datetime(created),
            "thread": reduce_thread_id(record.thread),
        }
        if record.exc_info is not None:
            obj["error"] = traceback.format_exception(*record.exc_info)[1:]
        try:
            obj["context"] = self.local.context
        except AttributeError:
            pass
        json.dump(obj, sys.stdout, sort_keys=True)
        sys.stdout.write("\n")
        sys.stdout.flush()

    def flush(self):
        sys.stdout.flush()


def setup(level: Union[str, int], structured: bool, config_path: str = None):
    """
    Makes stdout and stderr unicode friendly in case of misconfigured
    environments, initializes the logging, structured logging and
    enables colored logs if it is appropriate.

    :param level: The global logging level.
    :param structured: Output JSON logs to stdout.
    :param config_path: Path to a yaml file that configures the level of output of the loggers. \
                        Root logger level is set through the level argument and will override any \
                        root configuration found in the conf file.
    :return: None
    """
    global logs_are_structured
    logs_are_structured = structured

    if not isinstance(level, int):
        level = logging._nameToLevel[level]

    def ensure_utf8_stream(stream):
        if not isinstance(stream, io.StringIO) and hasattr(stream, "buffer"):
            stream = codecs.getwriter("utf-8")(stream.buffer)
            stream.encoding = "utf-8"
        return stream

    sys.stdout, sys.stderr = (ensure_utf8_stream(s)
                              for s in (sys.stdout, sys.stderr))

    # basicConfig is only called to make sure there is at least one handler for the root logger.
    # All the output level setting is down right afterwards.
    logging.basicConfig()
    logging.setLogRecordFactory(NumpyLogRecord)
    if config_path:
        with open(config_path) as fh:
            config = yaml.safe_load(fh)
        for key, val in config.items():
            logging.getLogger(key).setLevel(logging._nameToLevel.get(val, level))
    root = logging.getLogger()
    root.setLevel(level)

    if not structured:
        if not sys.stdin.closed and sys.stdout.isatty():
            handler = root.handlers[0]
            handler.setFormatter(AwesomeFormatter())
    else:
        root.handlers[0] = StructuredHandler(level)


def set_context(context):
    try:
        handler = logging.getLogger().handlers[0]
    except IndexError:
        # logging is not initialized
        return
    if not isinstance(handler, StructuredHandler):
        return
    handler.acquire()
    try:
        handler.local.context = context
    finally:
        handler.release()
