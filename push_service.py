# push_service.py
import requests
import json
from time import time
from utils import access_token, user_id, group_id, BOT_ID
from utils import handle_message 


def handshake():
    url = "https://push.groupme.com/faye"
    payload = [
        {
            "channel": "/meta/handshake",
            "version": "1.0",
            "supportedConnectionTypes": ["long-polling"],
            "id": "1"
        }
    ]
    headers = {"Content-Type": "application/json"}
    response = requests.post(url, json=payload, headers=headers)
    if response.status_code == 200:
        data = response.json()
        if data[0].get("successful", False):
            return data[0]["clientId"]
    return None

def subscribe_to_user_channel(client_id):
    url = "https://push.groupme.com/faye"
    payload = [
        {
            "channel": "/meta/subscribe",
            "clientId": client_id,
            "subscription": f"/user/{user_id}",
            "id": "2",
            "ext": {
                "access_token": access_token,
                "timestamp": int(time())
            }
        }
    ]
    headers = {"Content-Type": "application/json"}
    response = requests.post(url, json=payload, headers=headers)
    if response.status_code == 200:
        data = response.json()
        if data[0].get("successful", False):
            return True
    return False

def subscribe_to_group_channel(client_id):
    url = "https://push.groupme.com/faye"
    payload = [
        {
            "channel": "/meta/subscribe",
            "clientId": client_id,
            "subscription": f"/group/{group_id}",
            "id": "3",
            "ext": {
                "access_token": access_token,
                "timestamp": int(time())
            }
        }
    ]
    headers = {"Content-Type": "application/json"}
    response = requests.post(url, json=payload, headers=headers)
    if response.status_code == 200:
        data = response.json()
        if data[0].get("successful", False):
            return True
    return False

def poll_for_data(client_id):
    url = "https://push.groupme.com/faye"
    payload = [
        {
            "channel": "/meta/connect",
            "clientId": client_id,
            "connectionType": "long-polling",
            "id": "4"
        }
    ]
    headers = {"Content-Type": "application/json"}
    response = requests.post(url, json=payload, headers=headers)
    if response.status_code == 200:
        data = response.json()
        return data
    return None

def start_push_service():
    client_id = handshake()
    if client_id:
        if subscribe_to_user_channel(client_id):
            print("Subscribed to user channel.")
        if subscribe_to_group_channel(client_id):
            print("Subscribed to group channel.")
        while True:
            data = poll_for_data(client_id)
            if data:
                print("Received data:", json.dumps(data, indent=2))
                for item in data:
                    if item.get("channel") == f"/user/{user_id}":
                        message_data = item.get("data", {})
                        if message_data.get("type") == "line.create":
                            message_text = message_data.get("subject", {}).get("text", "")
                            msg_user_id = message_data.get("subject", {}).get("user_id", "")
                            msg_group_id = message_data.get("subject", {}).get("group_id", "")
                            message_id = message_data.get("subject", {}).get("id", "")
                            sender_id = message_data.get("sender_id", "")
                            handle_message(message_text, msg_user_id, msg_group_id, message_id, sender_id)