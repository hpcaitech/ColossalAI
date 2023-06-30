import os
from datetime import datetime, timedelta
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import pytz
import requests
import seaborn
from requests_toolbelt import MultipartEncoder


class Counter(dict):
    """
    Dataclass for a github contributor.

    Args:
        name (str): name of the contributor
        num_commits_this_week (int): number of commits made within one week
    """

    def record(self, item: str):
        if item in self:
            self[item] += 1
        else:
            self[item] = 1

    def to_sorted_list(self):
        data = [(key, value) for key, value in self.items()]
        data.sort(key=lambda x: x[1], reverse=True)
        return data


def get_utc_time_one_week_ago():
    """
    Get the UTC time one week ago.
    """
    now = datetime.utcnow()
    start_datetime = now - timedelta(days=7)
    return start_datetime


def datetime2str(dt):
    """
    Convert datetime to string in the format of YYYY-MM-DDTHH:MM:SSZ
    """
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def str2datetime(string):
    """
    Convert string in the format of YYYY-MM-DDTHH:MM:SSZ to datetime
    """
    return datetime.strptime(string, "%Y-%m-%dT%H:%M:%SZ")


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


def get_organization_repositories(github_token, organization_name) -> List[str]:
    """
    Retrieve the public repositories under the organization.
    """
    url = f"https://api.github.com/orgs/{organization_name}/repos?type=public"

    # prepare header
    headers = {
        'Authorization': f'Bearer {github_token}',
        'Accept': 'application/vnd.github+json',
        'X-GitHub-Api-Version': '2022-11-28'
    }

    res = requests.get(url, headers=headers).json()
    repo_list = []

    for item in res:
        repo_list.append(item['name'])
    return repo_list


def get_issue_pull_request_comments(github_token: str, org_name: str, repo_name: str, since: str) -> Dict[str, int]:
    """
    Retrieve the issue/PR comments made by our members in the last 7 days.

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
        comment_api = f'https://api.github.com/repos/{org_name}/{repo_name}/issues/comments?since={since}&page={page}'
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
                issue_api = f'https://api.github.com/repos/{org_name}/{repo_name}/issues/{issue_id}'
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


def get_discussion_comments(github_token: str, org_name: str, repo_name: str, since: str) -> Dict[str, int]:
    """
    Retrieve the discussion comments made by our members in the last 7 days.
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
            repository(owner: "{org_name}", name: "{repo_name}"){{
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
            repository(owner: "{org_name}", name: "{repo_name}"){{
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
                discussion_updated_at = str2datetime(discussion['updatedAt'])

                # check if the updatedAt is within the last 7 days
                # if yes, add it to discussion_numbers
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

    # get the discussion comments and replies made by our member
    user_engagement_count = {}
    for discussion_number in discussion_numbers:
        cursor = None
        num_per_request = 10

        while True:
            query = _generate_comment_reply_count_for_discussion(discussion_number, num_per_request, cursor)
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
                                # if yes, add it to discussion_numbers

                                reply_updated_at = datetime.strptime(reply['updatedAt'], "%Y-%m-%dT%H:%M:%SZ")
                                if reply_updated_at > since:
                                    member_name = reply['author']['login']
                                    if member_name in user_engagement_count:
                                        user_engagement_count[member_name] += 1
                                    else:
                                        user_engagement_count[member_name] = 1
    return user_engagement_count


def generate_user_engagement_leaderboard_image(github_token: str, org_name: str, repo_list: List[str], output_path: str) -> bool:
    """
    Generate the user engagement leaderboard image for stats within the last 7 days

    Args:
        github_token (str): GitHub access token for API calls
        output_path (str): the path to save the image
    """

    # request to the Github API to get the users who have replied the most in the last 7 days
    start_datetime = get_utc_time_one_week_ago()
    start_datetime_str = datetime2str(start_datetime)

    # get the issue/PR comments and discussion comment count
    total_engagement_count = {}

    def _update_count(counter):
        for name, count in counter.items():
            if name in total_engagement_count:
                total_engagement_count[name] += count
            else:
                total_engagement_count[name] = count


    for repo_name in repo_list:
        print(f"Fetching user engagement count for {repo_name}/{repo_name}")
        issue_pr_engagement_count = get_issue_pull_request_comments(github_token=github_token, org_name=org_name, repo_name=repo_name, since=start_datetime_str)
        discussion_engagement_count = get_discussion_comments(github_token=github_token, org_name=org_name, repo_name=repo_name, since=start_datetime)

        # update the total engagement count
        _update_count(issue_pr_engagement_count)
        _update_count(discussion_engagement_count)
        
    # prepare the data for plotting
    x = []
    y = []

    if len(total_engagement_count) > 0:
        ranking = []
        for name, count in total_engagement_count.items():
            ranking.append((name, count))

        ranking.sort(key=lambda x: x[1], reverse=True)

        for name, count in ranking:
            x.append(count)
            y.append(name)

        # plot the leaderboard
        xlabel = f"Number of Comments made (since {start_datetime_str})"
        ylabel = "Member"
        title = 'Active User Engagement Leaderboard'
        plot_bar_chart(x, y, xlabel=xlabel, ylabel=ylabel, title=title, output_path=output_path)
        return True
    else:
        return False


def generate_contributor_leaderboard_image(github_token, org_name, repo_list, output_path) -> bool:
    """
    Generate the contributor leaderboard image for stats within the last 7 days

    Args:
        github_token (str): GitHub access token for API calls
        output_path (str): the path to save the image
    """
    # request to the Github API to get the users who have contributed in the last 7 days
    headers = {
        'Authorization': f'Bearer {github_token}',
        'Accept': 'application/vnd.github+json',
        'X-GitHub-Api-Version': '2022-11-28'
    }

    counter = Counter()
    start_datetime = get_utc_time_one_week_ago()

    def _get_url(org_name, repo_name, page):
        return f'https://api.github.com/repos/{org_name}/{repo_name}/pulls?per_page=50&page={page}&state=closed'

    def _iterate_by_page(org_name, repo_name):
        page = 1
        stop = False

        while not stop:
            print(f"Fetching pull request data for {org_name}/{repo_name} - page{page}")
            url = _get_url(org_name, repo_name, page)

            while True:
                response = requests.get(url, headers=headers).json()

                if isinstance(response, list):
                    # sometimes the Github API returns nothing
                    # request again if the response is not a list
                    break
                print("Empty response, request again...")

            if len(response) == 0:
                # if the response is empty, stop
                stop = True
                break

            # count the pull request and author from response
            for pr_data in response:
                merged_at = pr_data['merged_at']
                author = pr_data['user']['login']

                if merged_at is None:
                    continue

                merge_datetime = str2datetime(merged_at)

                if merge_datetime < start_datetime:
                    # if we found a pull request that is merged before the start_datetime
                    # we stop
                    stop = True
                    break
                else:
                    # record the author1
                    counter.record(author)

            # next page
            page += 1

    for repo_name in repo_list:
        _iterate_by_page(org_name, repo_name)

    # convert unix timestamp to Beijing datetime
    bj_start_datetime = datetime.fromtimestamp(start_datetime.timestamp(), tz=pytz.timezone('Asia/Shanghai'))
    bj_start_datetime_str = datetime2str(bj_start_datetime)

    contribution_list = counter.to_sorted_list()

    # remove contributors who has zero commits
    author_list = [x[0] for x in contribution_list]
    num_commit_list = [x[1] for x in contribution_list]

    # plot
    if len(author_list) > 0:
        xlabel = f"Number of Pull Requests (since {bj_start_datetime_str})"
        ylabel = "Contributor"
        title = 'Active Contributor Leaderboard'
        plot_bar_chart(num_commit_list, author_list, xlabel=xlabel, ylabel=ylabel, title=title, output_path=output_path)
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
    ORG_NAME = "hpcaitech"

    # get all open source repositories
    REPO_LIST = get_organization_repositories(GITHUB_TOKEN, ORG_NAME)

    # generate images
    contrib_success = generate_contributor_leaderboard_image(GITHUB_TOKEN, ORG_NAME, REPO_LIST, CONTRIBUTOR_IMAGE_PATH)
    engagement_success = generate_user_engagement_leaderboard_image(GITHUB_TOKEN, ORG_NAME, REPO_LIST, USER_ENGAGEMENT_IMAGE_PATH)

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
- 开发贡献者测评标准为：本周由公司成员与社区在所有开源仓库提交的Pull Request次数
- 用户互动榜单测评标准为：本周由公司成员在非成员在所有开源仓库创建的issue/PR/discussion中回复的次数
"""

    send_message_to_lark(message, LARK_WEBHOOK_URL)

    # send contributor image to lark
    if contrib_success:
        send_image_to_lark(contributor_image_key, LARK_WEBHOOK_URL)
    else:
        send_message_to_lark("本周没有成员贡献PR，无榜单图片生成。", LARK_WEBHOOK_URL)

    # send user engagement image to lark
    if engagement_success:
        send_image_to_lark(user_engagement_image_key, LARK_WEBHOOK_URL)
    else:
        send_message_to_lark("本周没有成员互动，无榜单图片生成。", LARK_WEBHOOK_URL)
