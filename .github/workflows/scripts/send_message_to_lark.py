import argparse

import requests


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--message", type=str)
    parser.add_argument("-u", "--url", type=str)
    return parser.parse_args()


def send_message_to_lark(message, webhook_url):
    data = {"msg_type": "text", "content": {"text": message}}
    requests.post(webhook_url, json=data)


if __name__ == "__main__":
    args = parse_args()
    send_message_to_lark(args.message, args.url)
