"""Utils to work with lookout-sdk binary."""
import io
import logging
import os
import pathlib
import random
from shutil import copyfileobj
import socket
import subprocess
import sys
import tarfile
from urllib.error import HTTPError
from urllib.request import urlopen

from lookout.core.api.version import __version__ as binver


file = pathlib.Path(__file__).parent / "server"


def fetch():
    """
    Fetch corresponding lookout-sdk executable.
    """
    log = logging.getLogger("fetch")
    try:
        buffer = io.BytesIO()
        with urlopen("https://github.com/src-d/lookout/releases/download/"
                     "%s/lookout-sdk_%s_%s_amd64.tar.gz" % (binver, binver, sys.platform)
                     ) as response:
            copyfileobj(response, buffer)
        buffer.seek(0)
        with tarfile.open(fileobj=buffer, mode="r:gz") as tar:
            with file.open("wb") as fout:
                copyfileobj(tar.extractfile("lookout-sdk_linux_amd64/lookout-sdk"), fout)
        os.chmod(str(file), 0o775)
    except HTTPError as e:
        if e.code == 404:
            log.error("Release %s for %s platform is missing." % (binver, sys.platform))
        raise e from None
    except Exception as e:
        if file.exists():
            os.remove(str(file))
        raise e from None


def run(cmd: str, fr: str, to: str, port: int, git_dir: str=".", config_json: str=None) -> None:
    """
    Run lookout-sdk executable. If you do not have it please fetch first.

    :param cmd: Sub-command to run.
    :param fr: Corresponds to --from flag.
    :param to: Corresponds to --to flag.
    :param port: Running analyzer port on localhost.
    :param git_dir: Corresponds to --git-dir flag.
    :param config_json: Corresponds to --config-json flag.
    """
    command = [
        str(file), cmd, "-v", "ipv4://localhost:%d" % port,
        "--from", fr,
        "--to", to,
        "--git-dir", git_dir,
    ]
    if config_json:
        command.extend(("--config-json", config_json))
    subprocess.run(
        command,
        stdout=sys.stdout, stderr=sys.stderr, check=True)


def find_port(attempts: int = 100) -> int:
    """
    Find available port on localhost.

    :param attempts: Attempts number.
    :return: Founded port number.
    """
    while True:
        attempts -= 1
        if attempts == 0:
            raise ConnectionError("cannot find an open port")
        port = random.randint(1024, 32768)
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.connect(("localhost", port))
        except ConnectionRefusedError:
            return port
        finally:
            try:
                s.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            s.close()
