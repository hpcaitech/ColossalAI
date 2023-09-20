#!/usr/bin/env python
# coding: utf-8

import argparse
import os
import re

import requests

COMMIT_API = "https://api.github.com/repos/hpcaitech/ColossalAI/commits"
TAGS_API = "https://api.github.com/repos/hpcaitech/ColossalAI/tags"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, help="output path for the release draft", required=True)
    parser.add_argument("--version", type=str, help="current version to release", required=True)
    return parser.parse_args()


def get_latest_tag_commit(headers=None):
    res = requests.get(url=TAGS_API, headers=headers)
    data = res.json()
    commit_hash = data[0]["commit"]["sha"]
    version = data[0]["name"]
    return commit_hash, version


def get_commit_info(commit_hash, headers=None):
    api = f"{COMMIT_API}/{commit_hash}"
    res = requests.get(url=api, headers=headers)
    return res.json()


def get_all_commit_info(since, headers=None):
    page = 1
    results = []

    while True:
        api = f"{COMMIT_API}?since={since}&per_page=100&page={page}"
        resp = requests.get(url=api, headers=headers)
        data = resp.json()

        # exit when no more data
        if len(data) == 0:
            break

        results.extend(data)
        page += 1

    return results


def collate_release_info(commit_info_list):
    results = dict()
    pattern = pattern = r"\[.*\]"

    for commit_info in commit_info_list:
        author = commit_info["commit"]["author"]["name"]

        try:
            author_url = commit_info["author"]["url"]
        except:
            # author can be None
            author_url = None
        msg = commit_info["commit"]["message"]
        match = re.search(pattern, msg)

        if match:
            tag = match.group().lstrip("[").rstrip("]").capitalize()
            if tag not in results:
                results[tag] = []
            results[tag].append((msg, author, author_url))

    return results


def generate_release_post_markdown(current_version, last_version, release_info):
    text = []

    # add highlights
    highlights = "## What's Changed \n\n"
    text.append(highlights)

    # add items
    for k, v in release_info.items():
        topic = f"### {k} \n"
        text.append(topic)

        for msg, author, author_url in v:
            # only keep the first line
            msg = msg.split("\n")[0]

            if author_url:
                item = f"{msg} by [{author}]({author_url})\n"
            else:
                item = f"{msg} by {author}\n"
            text.append(f"- {item}")

        text.append("\n")

    # add full change log
    text.append(
        f"**Full Changelog**: https://github.com/hpcaitech/ColossalAI/compare/{current_version}...{last_version}"
    )

    return text


if __name__ == "__main__":
    args = parse_args()
    token = os.environ["GITHUB_API_TOKEN"]
    headers = {"Authorization": token}

    # get previous release tag
    last_release_commit, last_version = get_latest_tag_commit(headers)
    last_release_commit_info = get_commit_info(last_release_commit, headers=headers)
    last_release_date = last_release_commit_info["commit"]["author"]["date"]

    # get the commits since last release
    commit_info = get_all_commit_info(since=last_release_date, headers=headers)
    commit_info = commit_info[:-1]  # remove the release commit

    # collate into markdown
    release_info = collate_release_info(commit_info)
    markdown_text = generate_release_post_markdown(args.version, last_version, release_info)

    # write into a file
    with open(args.out, "w") as f:
        for line in markdown_text:
            f.write(line)
