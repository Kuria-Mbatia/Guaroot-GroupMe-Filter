import threading
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import json
import re
from collections import deque, defaultdict
from time import time
from spam_detection import classify_message
from cachetools import TTLCache
from functools import lru_cache
import redis
from redis.exceptions import RedisError
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration constants
access_token = ""
user_id = ""
group_id = ""
BOT_ID = ""
API_ROOT = 'https://api.groupme.com/v3/'
POST_URL = f"{API_ROOT}bots/post"

# Initialize session with retries
session = requests.Session()
retries = Retry(total=3, backoff_factor=0.1, status_forcelist=[500, 502, 503, 504])
session.mount('https://', HTTPAdapter(max_retries=retries, pool_connections=10, pool_maxsize=10))

# Cache configuration
MESSAGE_CACHE_SIZE = 50
message_cache = TTLCache(maxsize=1000, ttl=3600)  # Cache for 1 hour

# Rate limiting configuration
RATE_LIMIT_WINDOW = 60  # Time window in seconds
RATE_LIMIT_COUNT = 100  # Maximum messages per window

# Initialize Redis client
redis_enabled = False  # Set to True if you want to use Redis
redis_client = None

if redis_enabled:
    try:
        redis_client = redis.Redis(host='localhost', port=6379, db=0)
        redis_client.ping()  # Test the connection
        logger.info("Redis connected successfully")
    except (redis.ConnectionError, redis.exceptions.ConnectionError):
        logger.warning("Redis connection failed. Using in-memory rate limiting.")
        redis_client = None

# Keyword patterns for message filtering
selling_keywords = ['sell', 'selling', 'sale', 'sold', 'vending', 'trading', 'dealing', 'cheap', 'price', 'buying']
ticket_keywords = ['ticket', 'tickets', 'admission', 'pass', 'entry']
concert_keywords = ['concert', 'show', 'performance', 'gig', 'event']
flagged_words = ['dm', 'messag', 'direct', 'contact', 'essay writer', 'student paper assignments']
keyword_regex = re.compile(r'\b(' + '|'.join(selling_keywords + ticket_keywords + concert_keywords + flagged_words) + r')\b', re.IGNORECASE)

class APIBatcher:
    def __init__(self):
        self.queue = []
        self._lock = threading.Lock()
    
    def add_message(self, message):
        with self._lock:
            self.queue.append(message)
            if len(self.queue) >= 10:
                self.flush()
    
    def flush(self):
        with self._lock:
            if not self.queue:
                return
            try:
                response = session.post(f"{API_ROOT}/bulk", json={"messages": self.queue})
                response.raise_for_status()
            except requests.RequestException as e:
                logger.error(f"Failed to send batch messages: {e}")
            finally:
                self.queue.clear()

api_batcher = APIBatcher()

# Add these functions to utils.py

def update_bot_avatar(avatar_url):
    """Update the bot's avatar image"""
    try:
        # GroupMe API endpoint for updating a bot
        url = f"{API_ROOT}bots"
        
        # Get the current bot info first
        response = session.get(
            url,
            params={'token': access_token}
        )
        response.raise_for_status()
        
        bots = response.json()['response']
        our_bot = next((bot for bot in bots if bot['bot_id'] == BOT_ID), None)
        
        if our_bot:
            # Update the bot with the new avatar URL
            update_data = {
                "bot": {
                    "avatar_url": avatar_url,
                    "name": our_bot['name'],
                    "callback_url": our_bot['callback_url'],
                    "group_id": our_bot['group_id']
                }
            }
            
            response = session.post(
                f"{url}/{our_bot['bot_id']}", 
                json=update_data,
                params={'token': access_token}
            )
            response.raise_for_status()
            logger.info(f"Successfully updated bot avatar to: {avatar_url}")
            return True
    except Exception as e:
        logger.error(f"Failed to update bot avatar: {e}")
        return False

def ensure_bot_avatar():
    """Ensure the bot's avatar is set correctly"""
    # Replace this URL with your desired avatar image URL
    desired_avatar_url = "https://images.spr.so/cdn-cgi/imagedelivery/j42No7y-dcokJuNgXeA0ig/d30eb865-dd06-4e95-a1f4-8cbbef719090/Untitled_design_(8)-modified/w=750,quality=90,fit=scale-down"
    return update_bot_avatar(desired_avatar_url)

#Bot Startup routine
def initialize_bot():
    """Initialize bot settings"""
    logger.info("Initializing bot...")
    ensure_bot_avatar()
    send_message("Bot is online and ready to process messages!")


def get_user_cache(user_id):
    """Safely get or create a user's cache entry"""
    if user_id not in message_cache:
        message_cache[user_id] = deque(maxlen=MESSAGE_CACHE_SIZE)
    return message_cache[user_id]

def is_bot_message(sender_id, sender_type=None):
    """Check if a message is from a bot"""
    return sender_id == BOT_ID or sender_type == 'bot'

def send_message(message):
    """Send a message to the group"""
    data = {"bot_id": BOT_ID, "text": message}
    try:
        logger.info(f"Sending message: {message}")
        response = session.post(POST_URL, json=data)
        response.raise_for_status()
        logger.info("Message sent successfully")
    except requests.RequestException as e:
        logger.error(f"Failed to send message: {e}")

def get_memberships(group_id):
    """Get all members of a group"""
    try:
        response = session.get(
            f'{API_ROOT}groups/{group_id}',
            params={'token': access_token}
        )
        response.raise_for_status()
        return response.json()['response']['members']
    except requests.RequestException as e:
        logger.error(f"Failed to get memberships: {e}")
        return []

def get_membership_id(group_id, user_id):
    """Get a user's membership ID in a group"""
    memberships = get_memberships(group_id)
    for membership in memberships:
        if membership['user_id'] == user_id:
            return membership['id']
    return None

def remove_member(group_id, membership_id):
    """Remove a member from a group"""
    try:
        response = session.post(
            f'{API_ROOT}groups/{group_id}/members/{membership_id}/remove',
            params={'token': access_token}
        )
        response.raise_for_status()
        logger.info(f"Successfully removed member {membership_id} from group {group_id}")
        return True
    except requests.RequestException as e:
        logger.error(f"Failed to remove member: {e}")
        return False

def delete_message(group_id, message_id):
    """Delete a message from a group"""
    try:
        response = session.delete(
            f'{API_ROOT}conversations/{group_id}/messages/{message_id}',
            params={'token': access_token}
        )
        return response.status_code == 204
    except requests.RequestException as e:
        logger.error(f"Failed to delete message: {e}")
        return False

@lru_cache(maxsize=100)
def get_group_info(group_id):
    """Get information about a group"""
    try:
        response = session.get(
            f'{API_ROOT}groups/{group_id}',
            params={'token': access_token}
        )
        response.raise_for_status()
        return response.json()['response']
    except requests.RequestException as e:
        logger.error(f"Failed to get group info: {e}")
        return None

def is_admin_or_creator(group_id, user_id):
    """Check if a user is an admin or creator of a group"""
    group_info = get_group_info(group_id)
    if group_info:
        if group_info['creator_user_id'] == user_id:
            return True
        for member in group_info['members']:
            if member['user_id'] == user_id and member.get('roles', []) == ['admin']:
                return True
    return False

def is_duplicate_message(user_id, message):
    """Check if a message is a duplicate"""
    user_cache = get_user_cache(user_id)
    return any(cached_msg['text'] == message for cached_msg in user_cache)

def add_to_cache(user_id, message):
    """Add a message to the user's cache"""
    user_cache = get_user_cache(user_id)
    user_cache.append({'text': message, 'time': time()})

def is_spam(user_id, message_text, sender_id=None, sender_type=None):
    """Check if a message is spam"""
    if is_bot_message(sender_id, sender_type):
        return False
        
    try:
        user_cache = get_user_cache(user_id)
        spam_count = sum(keyword_regex.search(cached_msg['text']) is not None 
                        for cached_msg in user_cache)
        return spam_count > 4
    except Exception as e:
        logger.error(f"Error checking spam status: {e}")
        return False

def is_rate_limited(user_id):
    """Check if a user is rate limited"""
    if redis_client:
        try:
            key = f"rate_limit:{user_id}"
            current = redis_client.incr(key)
            if current == 1:
                redis_client.expire(key, RATE_LIMIT_WINDOW)
            return current > RATE_LIMIT_COUNT
        except RedisError as e:
            logger.error(f"Redis error in rate limiting: {e}")
            return False
    else:
        # Fallback to in-memory rate limiting
        current_time = time()
        counts = [t for t in user_message_counts[user_id] 
                 if current_time - t <= RATE_LIMIT_WINDOW]
        user_message_counts[user_id] = counts
        user_message_counts[user_id].append(current_time)
        return len(counts) >= RATE_LIMIT_COUNT

def get_flagged_words(message):
    """Get flagged words from a message"""
    words = re.findall(r'\b\w+\b', message.lower())
    flagged = []
    for word in words:
        if word in selling_keywords:
            flagged.append(('selling', word))
        elif word in ticket_keywords:
            flagged.append(('ticket', word))
        elif word in concert_keywords:
            flagged.append(('concert', word))
        elif word in flagged_words:
            flagged.append(('flagged', word))
    return flagged

from cachetools import TTLCache
processed_messages = TTLCache(maxsize=1000, ttl=300)  # Cache message IDs for 5 minutes

def handle_message(message_text, user_id, group_id, message_id, sender_id, sender_type='user'):
    """Main message handling function"""
    logger.info(f"Processing message: {message_text}")
    
    # Check if we've already processed this message
    if message_id in processed_messages:
        logger.info(f"Skipping already processed message {message_id}")
        return
        
    # Mark message as processed
    processed_messages[message_id] = True
    
    if message_text.strip() == "A message was deleted":
        logger.info("Ignoring deleted message notification")
        return
        
    if not is_bot_message(sender_id, sender_type):
        if is_rate_limited(user_id):
            logger.warning(f"Rate limit exceeded for user {user_id}")
            send_message("You're sending messages too quickly. Please slow down.")
            return
        
    try:
        spam_probability = classify_message(message_text)
        logger.info(f"Spam probability: {spam_probability:.2%}")
        
        should_remove = False
        
        if not is_bot_message(sender_id, sender_type):
            if spam_probability > 0.5 or is_spam(user_id, message_text, sender_id, sender_type):
                should_remove = True
                logger.warning(f"Message flagged as spam with probability {spam_probability:.2%}")
            elif keyword_regex.search(message_text):
                flagged_words = get_flagged_words(message_text)
                logger.warning("Message flagged based on keywords: " + 
                             ", ".join(f"{category}: {word}" for category, word in flagged_words))
                should_remove = True
        
        if should_remove:
            # Send notification and perform actions atomically
            actions_performed = False
            
            if delete_message(group_id, message_id):
                logger.info(f"Deleted message {message_id}")
                actions_performed = True
            
            if not is_admin_or_creator(group_id, user_id) and not is_bot_message(sender_id, sender_type):
                membership_id = get_membership_id(group_id, user_id)
                if membership_id and remove_member(group_id, membership_id):
                    logger.info(f"Removed user {user_id} from group")
                    actions_performed = True
            
            # Only send alert if actions were actually performed

            if actions_performed:
               logger.warning(f"[ALERT] This message has been flagged as spam with probability {spam_probability:.2%}")
                
        else:
            if not is_duplicate_message(user_id, message_text):
                add_to_cache(user_id, message_text)
                
    except Exception as e:
        logger.error(f"Error processing message: {e}", exc_info=True)

# Initialize in-memory rate limiting
user_message_counts = defaultdict(list)