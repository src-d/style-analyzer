import lzma
import re
import sys

from tqdm import tqdm

typosre = re.compile(
    r"((fix|correct)(|ed)\s+(|a\s+|the\s+)(typo|misprint)s?\s+.*(func|function|method|var|variable|cls|class|struct|identifier|attr|attribute|prop|property|name))|(^s/[^/]+/[^/]+)",  # noqa: E501
    re.IGNORECASE)
typosblackre = re.compile(r"filename|file name|\spath|\scomment", re.IGNORECASE)


with open("candidates.txt", "w") as fout:
    with open("messages.txt.xz", "rb") as xzfile:
        xzfile.seek(0, 2)
        with tqdm(total=xzfile.tell()) as progress:
            xzfile.seek(0, 0)
            with lzma.open(xzfile) as messages:
                extra = b""
                i = 0
                while True:
                    chunk = messages.read(1 << 18)
                    if len(chunk) != 1 << 18:
                        break
                    progress.n = xzfile.tell()
                    progress.update(0)
                    parts = chunk.split(b"\0")
                    parts[0] = extra + parts[0]
                    extra = parts[-1]
                    for part in parts[:-1]:
                        msg = part.decode("utf-8", errors="ignore")
                        if typosre.search(msg) and not typosblackre.search(msg):
                            fout.write(str(i) + "\n")
                        i += 1
