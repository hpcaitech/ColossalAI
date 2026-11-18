"""Fast-forward server Git metadata to a verified Mac bundle; never edit working files."""

import argparse
import subprocess


def git(*args):
    return subprocess.check_output(["git", *args], text=True).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle")
    parser.add_argument("commit")
    args = parser.parse_args()
    branch = "codex/ouro-2d4"
    assert git("branch", "--show-current") == branch
    subprocess.run(["git", "diff", "--cached", "--quiet"], check=True)
    old = git("rev-parse", "HEAD")
    subprocess.run(["git", "bundle", "verify", args.bundle], check=True)
    subprocess.run(["git", "fetch", "--no-tags", args.bundle, "refs/heads/" + branch], check=True)
    target = git("rev-parse", "FETCH_HEAD")
    assert target == args.commit
    subprocess.run(["git", "merge-base", "--is-ancestor", old, target], check=True)
    names = git("diff", "--name-only", old, target).splitlines()
    for name in names:
        assert git("hash-object", "--", name) == git("rev-parse", target + ":" + name), name
    # --mixed changes only HEAD/index. Existing unrelated working-tree edits,
    # executable-bit changes and untracked files remain exactly as they are.
    subprocess.run(["git", "reset", "--mixed", target], check=True)
    print(f"Verified {len(names)} files; advanced {old} -> {target}; working files preserved.")


if __name__ == "__main__":
    main()
