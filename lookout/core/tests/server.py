import io
import os
import pathlib
import random
from shutil import copyfileobj
import socket
import subprocess
import sys
import tarfile
from urllib.request import urlopen

from lookout.core.api.version import __version__ as binver


file = pathlib.Path(__file__).parent / "server"


def fetch():
    try:
        buffer = io.BytesIO()
        with urlopen("https://github.com/src-d/lookout/releases/download/"
                     "%s/lookout-sdk_%s_linux_amd64.tar.gz" % (binver, binver)) as response:
            copyfileobj(response, buffer)
        buffer.seek(0)
        with tarfile.open(fileobj=buffer, mode="r:gz") as tar:
            with file.open("wb") as fout:
                copyfileobj(tar.extractfile("lookout-sdk_linux_amd64/lookout-sdk"), fout)
        os.chmod(str(file), 0o775)
    except Exception as e:
        if file.exists():
            os.remove(str(file))
        raise e from None


def run(cmd: str, fr: str, to: str, port, git_dir="."):
    subprocess.run([str(file), cmd, "-v", "ipv4://localhost:%s" % port, "--from", fr, "--to", to,
                    "--git-dir", git_dir], stdout=sys.stdout, stderr=sys.stderr, check=True)


def find_port(attempts=100) -> int:
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
