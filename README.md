
<H1 align="center">Guaroot, Groupme Spam Classifier</H1>
<H3 align="center">"Root out the bad stuff🍂"</H3>
<p align="center">
  <img src="https://github.com/Kuria-Mbatia/Guaroot/blob/main/Guaroot%20Images/file%20(1).jpg" />
</p>

# GroupMe Bot

A spam-detecting GroupMe bot with machine learning capabilities. This bot helps moderate GroupMe chats by detecting spam, monitoring message rates, and managing user interactions.

<div align="center">

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.68.0+-00a393.svg)](https://fastapi.tiangolo.com)

</div>

## Features

- 🤖 ML-based spam detection
- 🚫 Rate limiting and flood protection
- 🔍 Keyword filtering and moderation
- 💾 Message caching
- 👥 Group management tools
- 📊 Activity monitoring

## Prerequisites

Before deploying the bot, ensure you have:

- Python 3.8 or higher
- A GroupMe account
- Your Bot ID from [dev.groupme.com](https://dev.groupme.com)
- Git installed

## Quick Start (Local Development)

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/groupme-bot.git
cd groupme-bot
```

2. **Set up virtual environment**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure environment variables**
```bash
# Create .env file
echo "BOT_ID=your_bot_id_here
GROUP_ID=your_group_id_here" > .env
```

5. **Run the bot**
```bash
python __main__.py
```

## Deployment Options

### Option 1: Local Deployment with LocalTunnel

1. **Install LocalTunnel**
```bash
npm install -g localtunnel
```

2. **Start your bot**
```bash
python __main__.py
```

3. **Create tunnel**
```bash
lt --port 5000
```

4. Update your bot's callback URL at dev.groupme.com with the LocalTunnel URL

### Option 2: Heroku Deployment

1. **Install Heroku CLI and login**
```bash
# Install Heroku CLI from https://devcenter.heroku.com/articles/heroku-cli
heroku login
```

2. **Create Heroku app**
```bash
heroku create your-bot-name
```

3. **Configure environment variables**
```bash
heroku config:set BOT_ID=your_bot_id
heroku config:set GROUP_ID=your_group_id
```

4. **Deploy**
```bash
git push heroku master
```

### Option 3: AWS Elastic Beanstalk

1. **Install EB CLI**
```bash
pip install awsebcli
```

2. **Initialize and deploy**
```bash
eb init -p python-3.8 groupme-bot
eb create groupme-bot-env
eb setenv BOT_ID=your_bot_id GROUP_ID=your_group_id
eb deploy
```

### Option 4: Google Cloud App Engine

1. **Install Google Cloud SDK and initialize**
```bash
gcloud init
```

2. **Create app.yaml**
```yaml
runtime: python38
env_variables:
  BOT_ID: "your_bot_id"
  GROUP_ID: "your_group_id"
```

3. **Deploy**
```bash
gcloud app deploy
```

### Option 5: Azure App Service

1. **Install Azure CLI and login**
```bash
az login
```

2. **Create and deploy**
```bash
az group create --name groupmebot-rg --location eastus
az appservice plan create --name groupmebot-plan --resource-group groupmebot-rg --sku FREE
az webapp create --name your-bot-name --resource-group groupmebot-rg --plan groupmebot-plan
az webapp config appsettings set --name your-bot-name --resource-group groupmebot-rg --settings BOT_ID=your_bot_id GROUP_ID=your_group_id
az webapp up --name your-bot-name --resource-group groupmebot-rg
```

## Configuration

### Spam Detection Settings
Edit `spam_detection.py` to adjust:
- Spam detection thresholds
- ML model parameters
- Training data sources

### Rate Limiting
In `utils.py`:
```python
RATE_LIMIT_WINDOW = 60  # Seconds
RATE_LIMIT_COUNT = 100  # Max messages per window
```

## Troubleshooting

### Common Issues

1. **ModuleNotFoundError**
```bash
pip install -r requirements.txt
```

2. **Connection Refused**
- Check if GroupMe API is accessible
- Verify bot token and ID
- Ensure port 5000 is available

3. **Rate Limiting Issues**
- Adjust `RATE_LIMIT_WINDOW` and `RATE_LIMIT_COUNT`
- Check GroupMe API limits

### Deployment-Specific Issues

**Heroku**
```bash
heroku logs --tail
```

**AWS**
```bash
eb logs
```

**Google Cloud**
```bash
gcloud app logs tail
```

**Azure**
- Check logs in Azure portal

## Contributing

1. Fork the repository
2. Create your feature branch
   ```bash
   git checkout -b feature/AmazingFeature
   ```
3. Commit your changes
   ```bash
   git commit -m 'Add some AmazingFeature'
   ```
4. Push to the branch
   ```bash
   git push origin feature/AmazingFeature
   ```
5. Open a Pull Request

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Acknowledgments

* [GroupMe API Documentation](https://dev.groupme.com/docs/v3)
* [FastAPI](https://fastapi.tiangolo.com)
* [scikit-learn](https://scikit-learn.org/)



