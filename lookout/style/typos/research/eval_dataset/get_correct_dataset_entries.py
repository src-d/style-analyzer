import csv
from pathlib import Path
import urllib.error
import urllib.request

from lookout.style.format.benchmarks.quality_report import handle_input_arg

template = "https://raw.githubusercontent.com/{user}/{repo}/{commit}/{path}"
github_template = "https://github.com/{user}/{repo}/blob//{commit}/{path}"

data_dir = Path(__file__).parents[2] / "benchmarks" / "data"
dataset = Path(__file__).parents[0] / "typos_dataset.csv.xz"
dataset = list(csv.DictReader(handle_input_arg(dataset)))
good_counter = 0
with open(str(data_dir / "commits_with_typo_filtered.csv"), "w") as f:
    writer = csv.DictWriter(
        f, "wrong_id,correct_id,file_fix,line,commit_fix,repo,commit_typo,file_typo".split(","))
    writer.writeheader()
    for i, row in enumerate(dataset, start=1):
        good_row = False
        row["line"] = int(row["line"])
        print("%d." % i, row["repo"], "Line #%d" % row["line"])
        user, repo = row["repo"].split("/")[-2:]
        for commit, file in (("commit_typo", "file_typo"),
                             ("commit_fix", "file_fix")):
            url = template.format(user=user, repo=repo, commit=row[commit], path=row[file])
            url_user = github_template.format(user=user, repo=repo, commit=row[commit],
                                              path=row[file])
            print(" ", commit, url_user)
            try:
                code = urllib.request.urlopen(url).read().decode('utf-8')
            except urllib.error.HTTPError as e:
                print("  ", e)
                continue
            lines = code.splitlines()
            try:
                print("   ", "wrong   id:", row["wrong_id"], "Exists in the line:",
                      row["wrong_id"] in lines[row["line"]])
                print("   ", "correct id:", row["correct_id"], "Exists in the line:",
                      row["correct_id"] in lines[row["line"]])
                print("   ", "Line: |%s|" % lines[row["line"]])
                if commit == "commit_typo":
                    good_row = row["wrong_id"] in lines[row["line"]]
            except IndexError:
                print("    Only %d lines. Requested line is #%d" % (len(lines), row["line"]))
        print("-" * 80)

        if good_row:
            writer.writerow(row)
            good_counter += 1

print("Good are %d / %d." % (good_counter, i))
