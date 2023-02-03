import os
from dataclasses import dataclass
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import pytz
import requests
import seaborn
from requests_toolbelt import MultipartEncoder


@dataclass
class Contributor:
    name: str
    num_commits_this_week: int


def generate_user_engagement_leaderboard_image(github_token, output_path):
    # request to the Github API to get the users who have replied the most in the last 7 days
    now = datetime.utcnow()
    start_datetime = now - timedelta(days=7)
    start_datetime_str = start_datetime.strftime("%Y-%m-%dT%H:%M:%SZ")

    # prepare header
    headers = {
        'Authorization': f'Bearer {github_token}',
        'Accept': 'application/vnd.github+json',
        'X-GitHub-Api-Version': '2022-11-28'
    }

    user_engagement_count = {}

    # do pagination to the API
    page = 1
    while True:
        comment_api = f'https://api.github.com/repos/hpcaitech/ColossalAI/issues/comments?since={start_datetime_str}&page={page}'
        comment_response = requests.get(comment_api, headers=headers).json()

        if len(comment_response) == 0:
            break
        else:
            for item in comment_response:
                comment_author_relationship = item['author_association']
                if comment_author_relationship != 'MEMBER':
                    # if the comment is not made by our member
                    # we don't count this comment towards user engagement
                    continue

                issue_id = item['issue_url'].split('/')[-1]
                issue_api = f'https://api.github.com/repos/hpcaitech/ColossalAI/issues/{issue_id}'
                issue_response = requests.get(issue_api, headers=headers).json()
                issue_author_relationship = issue_response['author_association']

                if issue_author_relationship != 'MEMBER':
                    # this means that the issue/PR is not created by our own people
                    # any comments in this issue/PR by our member will be counted towards the leaderboard
                    member_name = item['user']['login']

                    if member_name in user_engagement_count:
                        user_engagement_count[member_name] += 1
                    else:
                        user_engagement_count[member_name] = 1
            page += 1

    # plot the leaderboard
    x = []
    y = []

    for name, count in user_engagement_count.items():
        x.append(count)
        y.append(name)
    xticks = [str(v) for v in range(1, max(x) + 1)]
    seaborn.color_palette()
    fig = seaborn.barplot(x=x, y=y)
    fig.set(xlabel=f"Number of Comments made (since {start_datetime})",
            ylabel="Member",
            title='Active User Engagement Leaderboard')
    seaborn.despine()
    plt.tight_layout()
    plt.savefig(output_path, dpi=1200)


def generate_contributor_leaderboard_image(github_token, output_path):
    URL = 'https://api.github.com/repos/hpcaitech/ColossalAI/stats/contributors'
    headers = {
        'Authorization': f'Bearer {github_token}',
        'Accept': 'application/vnd.github+json',
        'X-GitHub-Api-Version': '2022-11-28'
    }
    response = requests.get(URL, headers=headers).json()

    contributor_list = []

    # convert unix timestamp to Beijing datetime
    start_timestamp = response[0]['weeks'][-1]['w']
    start_datetime = datetime.fromtimestamp(start_timestamp, tz=pytz.timezone('Asia/Shanghai'))

    # get number of commits for each contributor
    for item in response:
        num_commits_this_week = item['weeks'][-1]['c']
        name = item['author']['login']
        contributor = Contributor(name=name, num_commits_this_week=num_commits_this_week)
        contributor_list.append(contributor)

    # sort by number of commits
    contributor_list.sort(key=lambda x: x.num_commits_this_week, reverse=True)

    # remove contributors who has zero commits
    contributor_list = [x for x in contributor_list if x.num_commits_this_week > 0]

    # plot
    seaborn.color_palette()
    x = [x.num_commits_this_week for x in contributor_list]
    y = [x.name for x in contributor_list]
    fig = seaborn.barplot(x=x, y=y)
    fig.set(xlabel=f"Number of Commits (since {start_datetime})",
            ylabel="Contributor",
            title='Active Contributor Leaderboard')
    seaborn.despine()
    plt.tight_layout()
    plt.savefig(output_path, dpi=1200)


def upload_image_to_lark(lark_tenant_token, image_path):
    url = "https://open.feishu.cn/open-apis/im/v1/images"
    form = {'image_type': 'message', 'image': (open(image_path, 'rb'))}    # 需要替换具体的path
    multi_form = MultipartEncoder(form)
    headers = {
        'Authorization': f'Bearer {lark_tenant_token}',    ## 获取tenant_access_token, 需要替换为实际的token
    }
    headers['Content-Type'] = multi_form.content_type
    response = requests.request("POST", url, headers=headers, data=multi_form).json()
    return response['data']['image_key']


def generate_lark_tenant_access_token(app_id, app_secret):
    url = 'https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal'
    data = {'app_id': app_id, 'app_secret': app_secret}
    response = requests.post(url, json=data).json()
    return response['tenant_access_token']


def send_image_to_lark(image_key, webhook_url):
    data = {"msg_type": "image", "content": {"image_key": image_key}}
    requests.post(webhook_url, json=data)


def send_message_to_lark(message, webhook_url):
    data = {"msg_type": "text", "content": {"text": message}}
    requests.post(webhook_url, json=data)


if __name__ == '__main__':
    GITHUB_TOKEN = os.environ['GITHUB_TOKEN']
    CONTRIBUTOR_IMAGE_PATH = 'contributor_leaderboard.png'
    USER_ENGAGEMENT_IMAGE_PATH = 'engagement_leaderboard.png'

    # generate images
    # generate_contributor_leaderboard_image(GITHUB_TOKEN, CONTRIBUTOR_IMAGE_PATH)
    generate_user_engagement_leaderboard_image(GITHUB_TOKEN, USER_ENGAGEMENT_IMAGE_PATH)

    # upload images
    APP_ID = os.environ['LARK_APP_ID']
    APP_SECRET = os.environ['LARK_APP_SECRET']
    LARK_TENANT_TOKEN = generate_lark_tenant_access_token(app_id=APP_ID, app_secret=APP_SECRET)
    contributor_image_key = upload_image_to_lark(LARK_TENANT_TOKEN, CONTRIBUTOR_IMAGE_PATH)
    user_engagement_image_key = upload_image_to_lark(LARK_TENANT_TOKEN, USER_ENGAGEMENT_IMAGE_PATH)

    # send contributor image to lark
    LARK_WEBHOOK_URL = os.environ['LARK_WEBHOOK_URL']
    send_message_to_lark("本周的开发者贡献榜单出炉啦！", LARK_WEBHOOK_URL)
    send_image_to_lark(contributor_image_key, LARK_WEBHOOK_URL)

    # send user engagement image to lark
    send_message_to_lark("本周的开源社区互动榜单出炉啦！", LARK_WEBHOOK_URL)
    send_image_to_lark(user_engagement_image_key, LARK_WEBHOOK_URL)
