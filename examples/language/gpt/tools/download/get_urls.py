import datetime

import praw
import psaw
import tqdm

api = psaw.PushshiftAPI()

# all posts until the end of 2017
end_time = int(datetime.datetime(2018, 1, 1).timestamp())

query = api.search_submissions(before=end_time,
                               filter=['url', 'score'],
                               sort='desc',
                               score='>2',
                               is_self=False,
                               over_18=False)

with tqdm.tqdm() as pbar:
    # download links from submissions
    with open('urls.txt', 'w') as fh:
        for subm in query:
            url = subm.url

            # weird issue with psaw/pushshift that breaks score=">2"
            if subm.score < 3:
                continue
            #print(subm.score)
#            pbar.write(str(datetime.datetime.fromtimestamp(subm.created_utc)))
            pbar.update(1)
            fh.write(url + '\n')
        fh.flush()
