import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import pytz
import requests
import seaborn
from requests_toolbelt import MultipartEncoder


@dataclass
class Contributor:
    """
    Dataclass for a github contributor.

    Args:
        name (str): name of the contributor
        num_commits_this_week (int): number of commits made within one week
    """
    name: str
    num_commits_this_week: int


def plot_bar_chart(x: List[Any], y: List[Any], xlabel: str, ylabel: str, title: str, output_path: str) -> None:
    """
    This function is a utility to plot the bar charts.
    """
    plt.clf()
    seaborn.color_palette()
    fig = seaborn.barplot(x=x, y=y)
    fig.set(xlabel=xlabel, ylabel=ylabel, title=title)
    seaborn.despine()
    plt.tight_layout()
    plt.savefig(output_path, dpi=1200)


def get_issue_pull_request_comments(github_token: str, since: str) -> Dict[str, int]:
    """
    Retrive the issue/PR comments made by our members in the last 7 days.

    Args:
        github_token (str): GitHub access token for API calls
        since (str): the path parameter required by GitHub Restful APIs, in the format of YYYY-MM-DDTHH:MM:SSZ
    """
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
        comment_api = f'https://api.github.com/repos/hpcaitech/ColossalAI/issues/comments?since={since}&page={page}'
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
    return user_engagement_count


def get_discussion_comments(github_token, since) -> Dict[str, int]:
    """
    Retrive the discussion comments made by our members in the last 7 days.
    This is only available via the GitHub GraphQL API.

    Args:
        github_token (str): GitHub access token for API calls
        since (Datetime): the query parameter to determine whether the comment is made this week
    """

    # use graphql to get the discussions updated in the last 7 days
    def _generate_discussion_query(num, cursor: str = None):
        if cursor is None:
            offset_str = ""
        else:
            offset_str = f", after: \"{cursor}\""
        query = f"""
        {{
            repository(owner: "hpcaitech", name: "ColossalAI"){{
                discussions(first: {num} {offset_str}){{
                    edges {{
                        cursor
                        node{{
                            title
                            author{{
                                login
                            }}
                            number
                            authorAssociation
                            updatedAt
                        }}
                    }}
                }}
            }}
        }}
        """
        return query

    def _generate_comment_reply_count_for_discussion(discussion_number, num, cursor: str = None):
        # here we assume that each comment will not have more than 100 replies for simplicity
        # otherwise, we have to go through pagination for both comment and reply
        if cursor is None:
            offset_str = ""
        else:
            offset_str = f", before: \"{cursor}\""
        query = f"""
        {{
            repository(owner: "hpcaitech", name: "ColossalAI"){{
                discussion(number: {discussion_number}){{
                    title
                    comments(last: {num} {offset_str}){{
                        edges{{
                            cursor
                            node {{
                                author{{
                                    login
                                }}
                                updatedAt
                                authorAssociation
                                replies (last: 100) {{
                                edges {{
                                    node {{
                                        author {{
                                            login
                                        }}
                                        updatedAt
                                        authorAssociation
                                        }}
                                    }}
                                }}
                            }}
                        }}
                    }}
                }}
            }}
        }}
        """
        return query

    # a utility function to make call to Github GraphQL API
    def _call_graphql_api(query):
        headers = {"Authorization": f"Bearer {github_token}"}
        json_data = {'query': query}
        response = requests.post('https://api.github.com/graphql', json=json_data, headers=headers)
        data = response.json()
        return data

    # get the discussion numbers updated in the last 7 days
    discussion_numbers = []
    num_per_request = 10
    cursor = None
    while True:
        query = _generate_discussion_query(num_per_request, cursor)
        data = _call_graphql_api(query)
        found_discussion_out_of_time_range = False

        edges = data['data']['repository']['discussions']['edges']
        if len(edges) == 0:
            break
        else:
            # keep the discussion whose author is not a member
            for edge in edges:
                # print the discussion title
                discussion = edge['node']

                discussion_updated_at = datetime.strptime(discussion['updatedAt'], "%Y-%m-%dT%H:%M:%SZ")
                # check if the updatedAt is within the last 7 days
                # if yes, add it to dicussion_numbers
                if discussion_updated_at > since:
                    if discussion['authorAssociation'] != 'MEMBER':
                        discussion_numbers.append(discussion['number'])
                else:
                    found_discussion_out_of_time_range = True

        if found_discussion_out_of_time_range:
            break
        else:
            # update cursor
            cursor = edges[-1]['cursor']

    # get the dicussion comments and replies made by our member
    user_engagement_count = {}
    for dicussion_number in discussion_numbers:
        cursor = None
        num_per_request = 10

        while True:
            query = _generate_comment_reply_count_for_discussion(dicussion_number, num_per_request, cursor)
            data = _call_graphql_api(query)

            # get the comments
            edges = data['data']['repository']['discussion']['comments']['edges']

            # update the cursor
            if len(edges) == 0:
                break
            else:
                # update cursor for pagination
                cursor = edges[-1]['cursor']

                for edge in edges:
                    comment = edge['node']
                    if comment['authorAssociation'] == 'MEMBER':
                        # check if the updatedAt is within the last 7 days
                        # if yes, add it to user_engagement_count
                        comment_updated_at = datetime.strptime(comment['updatedAt'], "%Y-%m-%dT%H:%M:%SZ")
                        if comment_updated_at > since:
                            member_name = comment['author']['login']
                            if member_name in user_engagement_count:
                                user_engagement_count[member_name] += 1
                            else:
                                user_engagement_count[member_name] = 1

                    # get the replies
                    reply_edges = comment['replies']['edges']
                    if len(reply_edges) == 0:
                        continue
                    else:
                        for reply_edge in reply_edges:
                            reply = reply_edge['node']
                            if reply['authorAssociation'] == 'MEMBER':
                                # check if the updatedAt is within the last 7 days
                                # if yes, add it to dicussion_numbers
                                reply_updated_at = datetime.strptime(reply['updatedAt'], "%Y-%m-%dT%H:%M:%SZ")
                                if reply_updated_at > since:
                                    member_name = reply['author']['login']
                                    if member_name in user_engagement_count:
                                        user_engagement_count[member_name] += 1
                                    else:
                                        user_engagement_count[member_name] = 1
    return user_engagement_count


def generate_user_engagement_leaderboard_image(github_token: str, output_path: str) -> bool:
    """
    Generate the user engagement leaderboard image for stats within the last 7 days

    Args:
        github_token (str): GitHub access token for API calls
        output_path (str): the path to save the image
    """

    # request to the Github API to get the users who have replied the most in the last 7 days
    now = datetime.utcnow()
    start_datetime = now - timedelta(days=7)
    start_datetime_str = start_datetime.strftime("%Y-%m-%dT%H:%M:%SZ")

    # get the issue/PR comments and discussion comment count
    issue_pr_engagement_count = get_issue_pull_request_comments(github_token=github_token, since=start_datetime_str)
    discussion_engagement_count = get_discussion_comments(github_token=github_token, since=start_datetime)
    total_engagement_count = {}

    # update the total engagement count
    total_engagement_count.update(issue_pr_engagement_count)
    for name, count in discussion_engagement_count.items():
        if name in total_engagement_count:
            total_engagement_count[name] += count
        else:
            total_engagement_count[name] = count

    # prepare the data for plotting
    x = []
    y = []

    if len(total_engagement_count) > 0:
        for name, count in total_engagement_count.items():
            x.append(count)
            y.append(name)

        # use Shanghai time to display on the image
        start_datetime_str = datetime.now(pytz.timezone('Asia/Shanghai')).strftime("%Y-%m-%dT%H:%M:%SZ")

        # plot the leaderboard
        xlabel = f"Number of Comments made (since {start_datetime_str})"
        ylabel = "Member"
        title = 'Active User Engagement Leaderboard'
        plot_bar_chart(x, y, xlabel=xlabel, ylabel=ylabel, title=title, output_path=output_path)
        return True
    else:
        return False


def generate_contributor_leaderboard_image(github_token, output_path) -> bool:
    """
    Generate the contributor leaderboard image for stats within the last 7 days

    Args:
        github_token (str): GitHub access token for API calls
        output_path (str): the path to save the image
    """
    # request to the Github API to get the users who have contributed in the last 7 days
    URL = 'https://api.github.com/repos/hpcaitech/ColossalAI/stats/contributors'
    headers = {
        'Authorization': f'Bearer {github_token}',
        'Accept': 'application/vnd.github+json',
        'X-GitHub-Api-Version': '2022-11-28'
    }

    while True:
        response = requests.get(URL, headers=headers).json()

        if len(response) != 0:
            # sometimes the Github API returns empty response for unknown reason
            # request again if the response is empty
            break

    contributor_list = []

    # get number of commits for each contributor
    start_timestamp = None
    for item in response:
        num_commits_this_week = item['weeks'][-1]['c']
        name = item['author']['login']
        contributor = Contributor(name=name, num_commits_this_week=num_commits_this_week)
        contributor_list.append(contributor)

        # update start_timestamp
        start_timestamp = item['weeks'][-1]['w']

    # convert unix timestamp to Beijing datetime
    start_datetime = datetime.fromtimestamp(start_timestamp, tz=pytz.timezone('Asia/Shanghai'))
    start_datetime_str = start_datetime.strftime("%Y-%m-%dT%H:%M:%SZ")

    # sort by number of commits
    contributor_list.sort(key=lambda x: x.num_commits_this_week, reverse=True)

    # remove contributors who has zero commits
    contributor_list = [x for x in contributor_list if x.num_commits_this_week > 0]

    # prepare the data for plotting
    x = [x.num_commits_this_week for x in contributor_list]
    y = [x.name for x in contributor_list]

    # plot
    if len(x) > 0:
        xlabel = f"Number of Commits (since {start_datetime_str})"
        ylabel = "Contributor"
        title = 'Active Contributor Leaderboard'
        plot_bar_chart(x, y, xlabel=xlabel, ylabel=ylabel, title=title, output_path=output_path)
        return True
    else:
        return False


def upload_image_to_lark(lark_tenant_token: str, image_path: str) -> str:
    """
    Upload image to Lark and return the image key

    Args:
        lark_tenant_token (str): Lark tenant access token
        image_path (str): the path to the image to be uploaded
    """
    url = "https://open.feishu.cn/open-apis/im/v1/images"
    form = {'image_type': 'message', 'image': (open(image_path, 'rb'))}    # 需要替换具体的path
    multi_form = MultipartEncoder(form)
    headers = {
        'Authorization': f'Bearer {lark_tenant_token}',    ## 获取tenant_access_token, 需要替换为实际的token
    }
    headers['Content-Type'] = multi_form.content_type
    response = requests.request("POST", url, headers=headers, data=multi_form).json()
    return response['data']['image_key']


def generate_lark_tenant_access_token(app_id: str, app_secret: str) -> str:
    """
    Generate Lark tenant access token.

    Args:
        app_id (str): Lark app id
        app_secret (str): Lark app secret
    """
    url = 'https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal'
    data = {'app_id': app_id, 'app_secret': app_secret}
    response = requests.post(url, json=data).json()
    return response['tenant_access_token']


def send_image_to_lark(image_key: str, webhook_url: str) -> None:
    """
    Send image to Lark.

    Args:
        image_key (str): the image key returned by Lark
        webhook_url (str): the webhook url to send the image
    """
    data = {"msg_type": "image", "content": {"image_key": image_key}}
    requests.post(webhook_url, json=data)


def send_message_to_lark(message: str, webhook_url: str):
    """
    Send message to Lark.

    Args:
        message (str): the message to be sent
        webhook_url (str): the webhook url to send the message
    """
    data = {"msg_type": "text", "content": {"text": message}}
    requests.post(webhook_url, json=data)


if __name__ == '__main__':
    GITHUB_TOKEN = os.environ['GITHUB_TOKEN']
    CONTRIBUTOR_IMAGE_PATH = 'contributor_leaderboard.png'
    USER_ENGAGEMENT_IMAGE_PATH = 'engagement_leaderboard.png'

    # generate images
    contrib_success = generate_contributor_leaderboard_image(GITHUB_TOKEN, CONTRIBUTOR_IMAGE_PATH)
    engagement_success = generate_user_engagement_leaderboard_image(GITHUB_TOKEN, USER_ENGAGEMENT_IMAGE_PATH)

    # upload images
    APP_ID = os.environ['LARK_APP_ID']
    APP_SECRET = os.environ['LARK_APP_SECRET']
    LARK_TENANT_TOKEN = generate_lark_tenant_access_token(app_id=APP_ID, app_secret=APP_SECRET)
    contributor_image_key = upload_image_to_lark(LARK_TENANT_TOKEN, CONTRIBUTOR_IMAGE_PATH)
    user_engagement_image_key = upload_image_to_lark(LARK_TENANT_TOKEN, USER_ENGAGEMENT_IMAGE_PATH)

    # send message to lark
    LARK_WEBHOOK_URL = os.environ['LARK_WEBHOOK_URL']
    message = """本周的社区榜单出炉啦！
1. 开发贡献者榜单
2. 用户互动榜单

注：
- 开发贡献者测评标准为：本周由公司成员提交的commit次数
- 用户互动榜单测评标准为：本周由公司成员在非成员创建的issue/PR/discussion中回复的次数
"""

    send_message_to_lark(message, LARK_WEBHOOK_URL)

    # send contributor image to lark
    if contrib_success:
        send_image_to_lark(contributor_image_key, LARK_WEBHOOK_URL)
    else:
        send_message_to_lark("本周没有成员贡献commit，无榜单图片生成。", LARK_WEBHOOK_URL)

    # send user engagement image to lark
    if engagement_success:
        send_image_to_lark(user_engagement_image_key, LARK_WEBHOOK_URL)
    else:
        send_message_to_lark("本周没有成员互动，无榜单图片生成。", LARK_WEBHOOK_URL)
