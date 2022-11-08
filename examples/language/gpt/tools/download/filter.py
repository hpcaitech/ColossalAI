import re

import tldextract
import tqdm
from utils import linecount

# https://stackoverflow.com/questions/7160737/python-how-to-validate-a-url-in-python-malformed-or-not
url_regex = re.compile(
    r'^(?:http)s?://'    # http:// or https://
    r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'    #domain...
    r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'    # ...or ip
    r'(?::\d+)?'    # optional port
    r'(?:/?|[/?]\S+)$',
    re.IGNORECASE)

# domains that aren't scraper friendly. do not include subdomains!
exclude_domains = set([
    # image & video hosting sites
    'imgur.com',
    'redd.it',
    'instagram.com',
    'discord.gg',
    'gfycat.com',
    'giphy.com',
    'reddituploads.com',
    'redditmedia.com',
    'twimg.com',
    'sli.mg',
    'magaimg.net',
    'flickr.com',
    'imgflip.com',
    'youtube.com',
    'youtu.be',
    'youtubedoubler.com',
    'vimeo.com',
    'twitch.tv',
    'streamable.com',
    'bandcamp.com',
    'soundcloud.com',

    # not scraper friendly
    'reddit.com',
    'gyazo.com',
    'github.com',
    'xkcd.com',
    'twitter.com',
    'spotify.com',
    'itunes.apple.com',
    'facebook.com',
    'gunprime.com',
    'strawpoll.me',
    'voyagefusion.com',
    'rollingstone.com',
    'google.com',
    'timeanddate.com',
    'walmart.com',
    'roanoke.com',
    'spotrac.com',

    # original paper excluded wikipedia
    'wikipedia.org',

    # lots of top posts for this one
    'battleforthenet.com',
])

exclude_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.gifv', '.pdf', '.mp4', '.mp3', '.ogv', '.webm', '.doc',
                      '.docx', '.log', '.csv', '.dat', '.iso', '.bin', '.exe', '.apk', '.jar', '.app', '.ppt', '.pps',
                      '.pptx', '.xml', '.gz', '.xz', '.bz2', '.tgz', '.tar', '.zip', '.wma', '.mov', '.wmv', '.3gp',
                      '.svg', '.rar', '.wav', '.avi', '.7z')


def should_exclude(url):

    ext = tldextract.extract(url)
    domain = '.'.join([x for x in ext if x])
    basedomain = '.'.join(ext[-2:])

    # Ignore non-URLs
    if len(url) <= 8 or ' ' in url or re.match(url_regex, url) is None:
        return True

    # Ignore excluded domains
    if basedomain in exclude_domains or domain in exclude_domains:
        return True

    # Ignore case-insensitive matches for excluded extensions
    if url.lower().split('?')[0].endswith(exclude_extensions):
        return True

    return False


if __name__ == '__main__':
    url_file = 'urls.txt'
    filtered_file = 'urls-filtered.txt'

    with open(url_file) as urls, open(filtered_file, 'w') as out:
        url_len = linecount(url_file)
        print("URL file is", url_len, "URLs long.")
        url_set = set()
        for line in tqdm.tqdm(urls, total=url_len):
            if len(line.strip()) == 0:
                continue    # Skip whitespace-only lines
            line = line.strip().split()[0]    # Drop any components following whitespace
            if should_exclude(line):
                continue
            url_set.add(line)
        for line in tqdm.tqdm(url_set):
            out.write(line + '\n')
