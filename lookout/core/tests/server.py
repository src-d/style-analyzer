import io
import os
import pathlib
from shutil import copyfileobj
import subprocess
import sys
import tarfile
from urllib.request import urlopen


file = pathlib.Path(__file__).parent / "server"


def fetch():
    try:
        buffer = io.BytesIO()
        with urlopen("https://github.com/src-d/lookout/releases/download/"
                     "v0.1.0/lookout_sdk_v0.1.0_linux_amd64.tar.gz") as response:
            copyfileobj(response, buffer)
        buffer.seek(0)
        with tarfile.open(fileobj=buffer, mode="r:gz") as tar:
            with file.open("wb") as fout:
                copyfileobj(tar.extractfile("lookout_sdk_linux_amd64/lookout"), fout)
        os.chmod(str(file), 0o775)
    except Exception as e:
        if file.exists():
            os.remove(str(file))
        raise e from None


def run(cmd: str, fr: str, to: str, port):
    subprocess.run([str(file), cmd, "-v", "ipv4://localhost:%s" % port, "--from", fr, "--to", to],
                   stdout=sys.stdout, stderr=sys.stderr, check=True)
