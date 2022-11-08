import hashlib
import multiprocessing as mp
import os
import traceback

import newspaper
import tldextract
import tqdm
from filter import should_exclude

hash = hashlib.sha256

try:
    os.mkdir('data')
except FileExistsError:
    pass


def dl(url):
    url = url.strip()

    if should_exclude(url):
        return

    ext = tldextract.extract(url)
    domain = '.'.join([x for x in ext if x])

    fname = 'data/{}-{}.txt'.format(domain, hash(url.encode()).hexdigest())
    if os.path.isfile(fname):
        return
#    print('Downloading', url)
    try:
        article = newspaper.Article(url, fetch_images=False)
        article.download()
        article.parse()
    except newspaper.article.ArticleException:
        #        print('Dead link:', url)
        return


#        traceback.print_exc()

    text = article.text

    if text.strip() == '':
        #        print('Empty')
        return

    with open(fname, 'w') as out:
        out.write(text)

if __name__ == '__main__':
    p = mp.Pool(100)    # num of download threads
    with open('urls.txt') as fh:
        urls = list(fh)

        list(tqdm.tqdm(p.imap(dl, urls), total=len(urls)))
        print('Done!')
