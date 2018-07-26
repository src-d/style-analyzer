import codecs
import datetime
import io
import json
import logging
import re
import sys
import threading
from typing import Union

import xxhash


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


class ColorFormatter(logging.Formatter):
    """
    logging Formatter which prints messages with colors.
    """
    GREEN_MARKERS = [" ok", "ok:", "finished", "complete", "ready",
                     "done", "running", "success", "saved"]
    GREEN_RE = re.compile("|".join(GREEN_MARKERS))

    def formatMessage(self, record: logging.LogRecord):
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
                  "m%(levelname)s\033[0m:%(rthread):%(name)s:\033[" + text_color + \
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
        try:
            obj["context"] = self.local.context
        except AttributeError:
            pass
        json.dump(obj, sys.stdout, sort_keys=True)

    def flush(self):
        sys.stdout.flush()


def setup(level: Union[str, int], structured: bool):
    """
    Makes stdout and stderr unicode friendly in case of misconfigured
    environments, initializes the logging, structured logging and
    enables colored logs if it is appropriate.

    :param level: The global logging level.
    :param structured: Output JSON logs to stdout.
    :return: None
    """
    if not isinstance(level, int):
        level = logging._nameToLevel[level]

    def ensure_utf8_stream(stream):
        if not isinstance(stream, io.StringIO) and hasattr(stream, "buffer"):
            stream = codecs.getwriter("utf-8")(stream.buffer)
            stream.encoding = "utf-8"
        return stream

    sys.stdout, sys.stderr = (ensure_utf8_stream(s)
                              for s in (sys.stdout, sys.stderr))
    logging.basicConfig(level=level)
    root = logging.getLogger()
    # In some cases root has handlers and basicConfig() is a no-op
    root.setLevel(level)
    if not structured:
        if not sys.stdin.closed and sys.stdout.isatty():
            handler = root.handlers[0]
            handler.setFormatter(ColorFormatter())
    else:
        root.handlers[0] = StructuredHandler(level)


def set_context(context):
    handler = logging.getLogger().handlers[0]
    if not isinstance(handler, StructuredHandler):
        return
    handler.acquire()
    try:
        handler.local.context = context
    finally:
        handler.release()
