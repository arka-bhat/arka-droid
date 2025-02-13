# Arka's Droid

A Slack bot powered by mistral-small-24b LLM that generates contextual responses in channels. Built with Flask, Redis, and OpenRouter API.

## Features

-   🤖 Responds to mentions in Slack channels
-   💾 Maintains conversation context (5 recent messages)
-   📜 Command to show last 5 messages
-   🧠 Smart responses via OpenRouter API
-   ⚙️ Easy configuration and deployment

## Prerequisites

-   Slack Workspace with Bot Token (`SLACK_BOT_TOKEN`)
-   Redis server for conversation storage
-   OpenRouter API key (`OPENROUTER_API_KEY`)
-   Python 3.8+

## Installation

1. Clone the repository:

```bash
git clone https://github.com/arka-bhat/arka-droid.git
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Configure environment variables in `.env`:

```env
SLACK_BOT_TOKEN=your_token_here
OPENROUTER_API_KEY=your_key_here
```

4. Start the bot:

```bash
python app.py
```

The server will start on port 3000.

## Usage

-   @mention the bot to get responses (Example: `Hello @Arka's Droid, what is capital of US`)
-   Type `list my last 5 messages` to see recent conversation
-   Bot maintains context for each channel/thread
