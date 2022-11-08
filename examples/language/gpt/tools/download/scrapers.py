# Code taken in large part from https://github.com/jcpeterson/openwebtext

import time
import unicodedata

import bs4
import newspaper
from filter import should_exclude
from htmlmin import minify
from lxml.html.clean import Cleaner


def find_and_filter_tag(tag, soup):
    """tag specific filter logic"""

    candidates = soup.find_all(tag)
    candidates = [unicodedata.normalize("NFKD", x.string) for x in candidates if x.string is not None]

    if tag == "p":
        candidates = [y.strip() for y in candidates if len(y.split(" ")) >= 4]
        count = sum(len(y.split(" ")) for y in candidates)
    else:
        raise NotImplementedError

    return (candidates, count)


def raw_scraper(url, memoize):
    t1 = time.time()
    if should_exclude(url):
        # heuristic to make downloading faster
        return None, {
            "url": url,
            "scraper": "raw",
        }

    try:
        cleaner = Cleaner()
        cleaner.javascript = True
        cleaner.style = True
        article = newspaper.Article(url, fetch_images=False, memoize_articles=memoize)
        article.download()
        html = minify(article.html)
        html = cleaner.clean_html(html)
        article.parse()
    except:
        return None, {
            "url": url,
            "scraper": "raw",
        }
    if article.text == "":
        return None, {
            "url": url,
            "scraper": "raw",
        }

    metadata = {"url": url, "elapsed": time.time() - t1, "scraper": "raw"}
    return html, metadata


def newspaper_scraper(url, memoize):
    t1 = time.time()
    if should_exclude(url):
        # heuristic to make downloading faster
        return None, {
            "url": url,
            "scraper": "newspaper",
        }

    try:
        article = newspaper.Article(url, fetch_images=False, memoize_articles=memoize)
        article.download()
        article.parse()
        text = article.text
        count = len(text.split())
    except:
        return None, {
            "url": url,
            "scraper": "newspaper",
        }

    metadata = {
        "url": url,
        "word_count": count,
        "elapsed": time.time() - t1,
        "scraper": "newspaper",
    }
    return text, metadata


def bs4_scraper(url, memoize):
    t1 = time.time()
    if should_exclude(url):
        # heuristic to make downloading faster
        return None, {
            "url": url,
            "scraper": "bs4",
        }

    try:
        article = newspaper.Article(url, fetch_images=False, memoize_articles=memoize)
        article.download()
        html = article.html
        soup = bs4.BeautifulSoup(html, "lxml")
        text, count = find_and_filter_tag("p", soup)
        # DDB: keep text as a single string for consistency with
        # newspaper_scraper
        text = " ".join(text)
    except:
        return None, {
            "url": url,
            "scraper": "bs4",
        }

    metadata = {
        "url": url,
        "word_count": count,
        "elapsed": time.time() - t1,
        "scraper": "bs4",
    }
    return text, metadata
