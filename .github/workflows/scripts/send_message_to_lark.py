import sys
import requests

def send_message_to_lark(message, webhook_url):
    data = {"msg_type": "text", "content": {"text": message}}
    requests.post(webhook_url, json=data)

if __name__ == '__main__':
    message = sys.argv[1]
    webhook_url = sys.argv[2]
    send_message_to_lark(message, webhook_url)
