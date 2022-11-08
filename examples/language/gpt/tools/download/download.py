# Code taken in large part from https://github.com/jcpeterson/openwebtext

from __future__ import print_function

import argparse
import io
import json
import multiprocessing as mpl
import os
import os.path as op
import sqlite3
import tarfile
import time
import warnings
from glob import glob
from hashlib import sha256

import tldextract
from scrapers import bs4_scraper, newspaper_scraper, raw_scraper
# for backward compatibility
from six.moves.urllib.request import urlopen
from tqdm import tqdm
from utils import chunks, extract_month, linecount, mkdir

parser = argparse.ArgumentParser()
parser.add_argument("url_file", type=str)
parser.add_argument(
    "--save_uncompressed",
    action="store_true",
    default=False,
    help="whether to save the raw txt files to disk",
)
parser.add_argument(
    "--output",
    type=str,
    default='raw.json',
    help="where to save the output json",
)
parser.add_argument(
    "--output_dir",
    type=str,
    default="scraped",
    help="which folder in the working directory to use for output",
)
parser.add_argument(
    "--n_procs",
    type=int,
    default=10,
    help="how many processes (cores) to use for parallel scraping",
)
parser.add_argument(
    "--timeout",
    type=int,
    default=-1,
    help="maximum scrape time for a single URL; -1 means no limit",
)
parser.add_argument(
    "--max_urls",
    type=int,
    default=-1,
    help="maximum # of URLs to scrape; mostly for debugging",
)
parser.add_argument(
    "--chunk_size",
    type=int,
    default=100,
    help="how many URLs to scrape before saving to archive",
)
parser.add_argument(
    "--scraper",
    type=str,
    default="newspaper",
    choices=["raw", "bs4", "newspaper"],
    help="which text/content scraper to use; raw is html",
)
parser.add_argument(
    "--compress",
    action="store_true",
    default=False,
    help="whether to output scraped content as compressed archives",
)
parser.add_argument(
    "--compress_fmt",
    type=str,
    default="xz",
    choices=["xz", "bz2", "gz"],
    help="which archive format to use",
)
parser.add_argument(
    "--scraper_memoize",
    action="store_true",
    default=False,
    help="whether to use cache for newspaper",
)
parser.add_argument(
    "--show_warnings",
    action="store_true",
    default=False,
    help="whether to show warnings in general during scraping",
)
parser.add_argument(
    "--sqlite_meta",
    action="store_true",
    default=True,
    help="whether to use sqlite for storing meta. if false, json will be used instead",
)
args = parser.parse_args()

if not args.show_warnings:
    # avoid lots of datetime warnings
    warnings.filterwarnings("ignore")


def load_urls(fh, max_urls=-1):
    url_entries = enumerate(fh)
    if max_urls != -1:
        url_entries = list(url_entries)[:max_urls]
    return url_entries


def vet_link(link):
    # check if server responds with non-200 status code or link points to a
    # non-html file
    link_type, link_status = "", -1
    try:
        info = urlopen(link)
        link_type = info.headers["Content-Type"]
        link_status = info.status
    except:
        pass

    # we want "text/html" only!
    is_good_link = False
    if "text/html" in link_type and link_status == 200:
        is_good_link = True

    return is_good_link, link_type


def download(url_entry,
             scraper=args.scraper,
             save_uncompressed=args.save_uncompressed,
             memoize=args.scraper_memoize,
             arch_meta=not args.sqlite_meta):

    uid, url = url_entry
    url = url.strip()
    fid = "{:07d}-{}".format(uid, sha256(url.encode()).hexdigest())

    data_dir = mkdir(op.join(args.output_dir, "data"))
    text_fp = op.join(data_dir, "{}.txt".format(fid))

    if arch_meta:
        meta_dir = mkdir(op.join(args.output_dir, "meta"))
        meta_fp = op.join(meta_dir, "{}.json".format(fid))

    # already downloaded!
    if op.exists(text_fp):
        return

    # is_good_link, link_type = vet_link(url)
    # if not is_good_link:
    #     return

    if scraper == "bs4":
        scrape = bs4_scraper
    elif scraper == "newspaper":
        scrape = newspaper_scraper
    elif scraper == "raw":
        scrape = raw_scraper

    text, meta = scrape(url, memoize)

    ext = tldextract.extract(url)
    domain = '.'.join([x for x in ext if x])
    meta["domain"] = domain

    if text is None or text.strip() == "":
        return ("", meta, fid, uid)

    if save_uncompressed:
        with open(text_fp, "w") as out:
            out.write(text)
        if arch_meta:
            with open(meta_fp, "w") as out:
                json.dump(meta, out)

    return (text, meta, fid, uid)


def archive_chunk(cid, cdata, out_dir, fmt, arch_meta):
    mkdir(out_dir)
    texts, metas, fids, uids = zip(*cdata)

    data_tar = op.join(out_dir, "{}_data.{}".format(cid, fmt))
    if arch_meta:
        meta_tar = op.join(out_dir, "{}_meta.{}".format(cid, fmt))
        tar_fps, texts, exts = [data_tar, meta_tar], [texts, metas], ["txt", "json"]
    else:
        tar_fps, texts, exts = [data_tar], [texts], ["txt"]

    doc_count = 0
    docs_counted = False
    for tar_fp, txts, ext in zip(tar_fps, texts, exts):
        with tarfile.open(tar_fp, "w:" + fmt) as tar:
            for f, fid in zip(txts, fids):
                if f == "":
                    continue
                else:
                    if not docs_counted:
                        doc_count += 1

                if ext == "json":
                    f = json.dumps(f)

                f = f.encode("utf-8")
                t = tarfile.TarInfo("{}.{}".format(fid, ext))
                t.size = len(f)
                tar.addfile(t, io.BytesIO(f))
        docs_counted = True

    return doc_count


def load_state(url_file):
    ckptfile = url_file + '.ckpt'
    if op.exists(ckptfile):
        with open(ckptfile) as fp:
            r = fp.read()
            if r == '':
                return 0
            else:
                return int(r)
    else:
        return 0


def save_state(url_file, cid):
    ckptfile = url_file + '.ckpt'
    with open(ckptfile, 'w') as fp:
        fp.write(str(cid))


def sqlite_conn():
    conn = sqlite3.connect('metadata.db')
    conn.execute('''
    CREATE TABLE IF NOT EXISTS metadata (
        fid char(64) not null primary key,
        url varchar(2048) not null,
        domain varchar(255) not null,
        word_count int null,
        elapsed int null,
        scraper varchar(255) not null,
        success boolean not null
    );
    ''')
    conn.execute('''
    CREATE INDEX IF NOT EXISTS ix_meta_url ON metadata(url);
    ''')
    conn.execute('''
    CREATE INDEX IF NOT EXISTS ix_meta_domain ON metadata(domain);
    ''')

    return conn


if __name__ == "__main__":
    if args.sqlite_meta:
        conn = sqlite_conn()
        cur = conn.cursor()

    start_elem = load_state(args.url_file)
    start_chnk = start_elem // args.chunk_size

    f_json = open(args.output, "w")

    # URLs we haven't scraped yet (if first run, all URLs in file)
    with open(args.url_file) as fh:
        url_entries = load_urls(fh, args.max_urls)

        pool = mpl.Pool(args.n_procs)
        total = linecount(args.url_file) // args.chunk_size
        print('Total chunks: ', total)
        chunk_iterator = tqdm(enumerate(chunks(url_entries, args.chunk_size, start_elem)), total=total)

        # display already-downloaded chunks on progress bar
        chunk_iterator.update(start_chnk)

        # process one "chunk" of args.chunk_size URLs at a time
        for i, chunk in chunk_iterator:
            cid = start_chnk + i + 1

            tqdm.write("Downloading chunk {}".format(cid))
            t1 = time.time()

            if args.timeout > 0:
                # imap as iterator allows .next() w/ timeout.
                # ordered version doesn't seem to work correctly.
                # for some reason, you CANNOT track j or chunk[j] in the loop,
                # so don't add anything else to the loop below!
                # confusingly, chunksize below is unrelated to our chunk_size
                chunk_iter = pool.imap_unordered(download, chunk, chunksize=1)
                cdata = []
                for j in range(len(chunk)):
                    try:
                        result = chunk_iter.next(timeout=args.timeout)
                        cdata.append(result)
                    except mpl.TimeoutError:
                        tqdm.write("   --- Timeout Error ---   ")
            else:
                cdata = list(pool.imap(download, chunk, chunksize=1))

            tqdm.write("{} / {} downloads timed out".format(len(chunk) - len(cdata), len(chunk)))
            tqdm.write("Chunk time: {} seconds".format(time.time() - t1))

            # write metadata to sqlite
            if args.sqlite_meta:
                for text, meta, fid, _ in filter(lambda x: x, cdata):
                    if text:
                        params = (fid, meta["url"], meta["domain"], meta["elapsed"], meta["word_count"],
                                  meta["scraper"], True)
                    else:
                        params = (fid, meta["url"], meta["domain"], None, None, meta["scraper"], False)
                    cur.execute(
                        "insert or ignore into metadata (fid, url, domain, elapsed, word_count, scraper, success) values (?, ?, ?, ?, ?, ?, ?)",
                        params)
                conn.commit()

            dump_chunk = []
            for text, meta, fid, _ in filter(lambda x: x, cdata):
                if text:
                    line_json = {"text": text, "url": meta["url"]}
                    dump_chunk.append(json.dumps(line_json) + '\n')
            f_json.writelines(dump_chunk)

            # archive and save this chunk to file
            if args.compress:
                tqdm.write("Compressing...")
                t2 = time.time()
                count = archive_chunk(cid, cdata, args.output_dir, args.compress_fmt, not args.sqlite_meta)
                tqdm.write("Archive created in {} seconds".format(time.time() - t2))
            tqdm.write("{} out of {} URLs yielded content\n".format(len(list(filter(lambda x: x and x[0], cdata))),
                                                                    len(chunk)))

            save_state(args.url_file, cid * args.chunk_size)
        f_json.close()
        print("Done!")
