import csv
import gzip
import re
import sys

from tqdm import tqdm


# >>> include s/xxx/yyy/

typosre = re.compile(r"(fix|correct)(|ed)\s+(|a\s+|the\s+)(typo|misprint)s?\s+.*(func|function|method|var|variable|cls|class|struct|identifier|attr|attribute|prop|property|name)")
typosblackre = re.compile(r"(filename|file name|\spath|\scomment)")
dates = input().split(" ")

typos = []
for date in tqdm(dates):
    with gzip.open("updates/messages-%s.txt.gz" % date) as msgfile:
        messages = msgfile.read().split(b"\0")
    with gzip.open("updates/repos-%s.txt.gz" % date) as repofile:
        repos = repofile.read().split(b"\0")
    with open("updates/commits-%s.bin" % date, "rb") as fin:
        commits = []
        while True:
            buf = fin.read(20)
            if len(buf) != 20:
                break
            commits.append(buf.hex())
    if messages[-1] == b"":
        messages = messages[:-1]
    if repos[-1] == b"":
        repos = repos[:-1]
    assert len(messages) == len(repos)
    assert len(repos) == len(commits)
    for msg, repo, commit in zip(messages, repos, commits):
        msg = msg.decode(errors="ignore")
        lmsg = msg.lower()
        if typosre.search(lmsg) and not typosblackre.search(lmsg):
            typos.append((repo.decode(), commit, msg))

typos.sort()
with open(sys.argv[1], "w") as fout:
    writer = csv.writer(fout)
    for t in typos:
        writer.writerow(t)
