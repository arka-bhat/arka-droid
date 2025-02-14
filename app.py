import hashlib
import hmac
import os
import json
import time
from typing import List, Optional, Dict

import requests
from flask import Flask, jsonify, request
import redis
from dotenv import load_dotenv
from slack_bolt import App
from slack_bolt.adapter.flask import SlackRequestHandler

# Load environment variables
load_dotenv()

model = "mistralai/mistral-small-24b-instruct-2501:free"
# Initialize Slack app
app = App(token=os.environ["SLACK_BOT_TOKEN"])
SLACK_SIGNING_SECRET = os.environ["SLACK_SIGNING_SECRET"]
SLACK_CLIENT_ID = os.getenv("SLACK_CLIENT_ID")
SLACK_CLIENT_SECRET = os.getenv("SLACK_CLIENT_SECRET")

# Initialize Redis client
redis_client = redis.Redis(
    host=os.environ.get("REDIS_HOST", "localhost"),
    port=int(os.environ.get("REDIS_PORT", 6379)),
    password=os.environ.get("REDIS_PASSWORD", ""),
    decode_responses=True
)

CONVERSATION_TTL = 24 * 60 * 60  # 24 hours in seconds

class Message:
    """Represents a chat message."""
    def __init__(
        self,
        text: str,
        channel: str,
        thread_ts: Optional[str] = None,
        timestamp: Optional[float] = None
    ):
        self.text = text
        self.channel = channel
        self.thread_ts = thread_ts
        self.timestamp = timestamp or time.time()

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "channel": self.channel,
            "thread_ts": self.thread_ts,
            "timestamp": self.timestamp
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Message':
        return cls(
            text=data["text"],
            channel=data["channel"],
            thread_ts=data["thread_ts"],
            timestamp=data["timestamp"]
        )

class OpenRouterClient:
    """Client for interacting with OpenRouter API."""
    def __init__(self, api_key: str, site_url: str, site_name: str):
        self.base_url = "https://openrouter.ai/api/v1"
        self.api_key = api_key
        self.site_url = site_url
        self.site_name = site_name
        self.model = model  # Default model

    def create_chat_completion(self, messages: List[Dict[str, str]]) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": self.site_url,
            "X-Title": self.site_name,
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "messages": messages
        }

        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=payload
        )
        response.raise_for_status()

        return response.json()["choices"][0]["message"]["content"]

class ChatBot:
    """Main bot class handling message processing and responses."""
    def __init__(self, redis_client: redis.Redis, llm_client: OpenRouterClient):
        self.redis = redis_client
        self.llm = llm_client

    def _get_conversation_key(self, channel_id: str, thread_ts: Optional[str] = None) -> str:
        if thread_ts:
            return f"conversation:{channel_id}:{thread_ts}"
        return f"conversation:{channel_id}"

    def store_message(self, message: Message) -> None:
        key = self._get_conversation_key(message.channel, message.thread_ts)
        
        message_json = json.dumps(message.to_dict())
        pipe = self.redis.pipeline()
        
        pipe.lpush(key, message_json)
        pipe.ltrim(key, 0, 4)
        pipe.expire(key, CONVERSATION_TTL)
        
        pipe.execute()

    def store_user_text(self, text: str) -> None:
        key = "user_text"
        pipe = self.redis.pipeline()
        
        pipe.lpush(key, text)
        pipe.ltrim(key, 0, 4)
        pipe.expire(key, CONVERSATION_TTL)
        
        pipe.execute()

    def get_last_messages(self, channel_id: str, thread_ts: Optional[str] = None) -> List[Message]:
        key = self._get_conversation_key(channel_id, thread_ts)
        messages_json = self.redis.lrange(key, 0, 4)
        
        return [Message.from_dict(json.loads(msg_json)) for msg_json in messages_json]
    
    def get_last_user_texts(self) -> List[str]:
        key = "user_text"
        user_texts = self.redis.lrange(key, 0, 4)

        return user_texts

    def prepare_conversation_context(self, messages: List[Message]) -> List[Dict[str, str]]:
        formatted_messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant in a Slack channel. Stick to short but helpful messages"
            }
        ]
        
        for msg in reversed(messages):
            formatted_messages.append({
                "role": "user",
                "content": msg.text
            })
        
        return formatted_messages

    def get_llm_response(self, messages: List[Dict[str, str]]) -> str:
        try:
            return self.llm.create_chat_completion(messages)
        except requests.exceptions.RequestException as e:
            print(f"Error getting LLM response: {e}")
            raise

def is_bot_mentioned(message_text: str, bot_user_id: str) -> bool:
    return f"<@{bot_user_id}>" in message_text

llm_client = OpenRouterClient(
    api_key=os.environ["OPENROUTER_API_KEY"],
    site_url=os.environ["SITE_URL"],
    site_name=os.environ["SITE_NAME"]
)
bot = ChatBot(redis_client, llm_client)

# Flask setup
flask_app = Flask(__name__)
handler = SlackRequestHandler(app)

def verify_slack_request(req):
    """
    Verifies Slack's request signature.
    """
    slack_signature = req.headers.get('X-Slack-Signature')
    slack_request_timestamp = req.headers.get('X-Slack-Request-Timestamp')

    # Ensure the request is not too old (Slack allows a 5-minute window)
    if abs(time.time() - int(slack_request_timestamp)) > 60 * 5:
        raise ValueError("Request timestamp is too old")

    # Create the base string from the request
    sig_basestring = f"v0:{slack_request_timestamp}:{req.get_data().decode('utf-8')}"

    # Create the expected signature using your signing secret
    secret = bytes(SLACK_SIGNING_SECRET, 'utf-8')
    sig_hash = "v0=" + hmac.new(secret, sig_basestring.encode('utf-8'), hashlib.sha256).hexdigest()

    # Compare the computed signature with Slack's signature
    if not hmac.compare_digest(sig_hash, slack_signature):
        raise ValueError("Invalid signature")

    return True

def get_ngrok_url():
    ngrok_info = requests.get("http://localhost:4040/api/tunnels").json()
    return ngrok_info['tunnels'][0]['public_url']

SLACK_REDIRECT_URI = get_ngrok_url() + "/slack/oauth/callback"
print(f"slack redirect URI: {SLACK_REDIRECT_URI}")

@flask_app.route("/slack/oauth/callback")
def oauth_callback():
    # Step 1: Get the `code` from Slack's callback
    code = request.args.get('code')
    if not code:
        return "Error: Missing code", 400

    # Step 2: Exchange the code for an access token
    token_url = "https://slack.com/api/oauth.v2.access"
    data = {
        'client_id': SLACK_CLIENT_ID,
        'client_secret': SLACK_CLIENT_SECRET,
        'code': code,
        'redirect_uri': SLACK_REDIRECT_URI
    }
    
    # Make the request to Slack to exchange the code for a token
    response = requests.post(token_url, data=data)
    response_data = response.json()

    if response_data.get("ok"):
        # Successfully received access token
        access_token = response_data["access_token"]
        # Save the access token (e.g., in a database or session) for making Slack API calls
        return "Slack bot installed successfully!"

    else:
        # Handle errors (e.g., invalid code or denied permissions)
        error_msg = response_data.get("error", "Unknown error")
        return f"Error: {error_msg}", 400
    
@flask_app.route("/slack/events", methods=["POST"])
def slack_events():
    try:
        # Verify the request
        verify_slack_request(request)

        # Pass the request to the SlackRequestHandler for further processing
        return handler.handle(request)
    
    except ValueError as e:
        print(f"Request verification failed: {e}")
        return jsonify({"error": "Unauthorized"}), 401

@app.event("message")
def handle_message(event, say, client):
    if event.get("bot_id"):
        return

    bot_user_id = client.auth_test()["user_id"]

    if is_bot_mentioned(event["text"], bot_user_id):
        if "list my last 5 messages" in event["text"].lower():
            try:
                user_texts = bot.get_last_user_texts()
                formatted_messages = "\n".join([f"{i + 1}. {text}" for i, text in enumerate(user_texts)])

                thread_ts = event.get("thread_ts", event["ts"])
                say(
                    text=f"Here are the last 5 user messages:\n{formatted_messages}",
                    thread_ts=thread_ts
                )
            except Exception as e:
                print(f"Error fetching last 5 messages: {str(e)}")
                say(
                    text="Sorry, I encountered an issue fetching the last 5 messages.",
                    thread_ts=event.get("thread_ts", event["ts"])
                )
        else:
            try:
                message = Message(
                    text=event["text"],
                    channel=event["channel"],
                    thread_ts=event.get("thread_ts")
                )
                bot.store_message(message)
                bot.store_user_text(event["text"])

                messages = bot.get_last_messages(event["channel"], event.get("thread_ts"))
                conversation_messages = bot.prepare_conversation_context(messages)

                response = bot.get_llm_response(conversation_messages)

                thread_ts = event.get("thread_ts", event["ts"])
                say(
                    text=response,
                    thread_ts=thread_ts
                )
            except Exception as e:
                print(f"Error handling message: {str(e)}")
                say(
                    text="Sorry, there was an issue in processing your request. Please try again after some time",
                    thread_ts=event.get("thread_ts", event["ts"])
                )

@app.event("app_mention")
def handle_app_mention(event, say, client):
    if event.get("bot_id"):
        return

    bot_user_id = client.auth_test()["user_id"]

    try:
        message = Message(
            text=event["text"],
            channel=event["channel"],
            thread_ts=event.get("thread_ts")
        )
        bot.store_message(message)

        messages = bot.get_last_messages(event["channel"], event.get("thread_ts"))
        conversation_messages = bot.prepare_conversation_context(messages)

        response = bot.get_llm_response(conversation_messages)

        thread_ts = event.get("thread_ts", event["ts"])
        say(
            text=response,
            thread_ts=thread_ts
        )

    except Exception as e:
        print(f"Error handling app_mention: {str(e)}")
        say(
            text="Sorry, there was an issue in processing your request. Please try again after some time",
            thread_ts=event.get("thread_ts", event["ts"])
        )

if __name__ == "__main__":
    flask_app.run(host='0.0.0.0',port=3000)
