from django.conf import settings
import logging
import sys

chatbotPath = "/".join(settings.BASE_DIR.split('/')[:-1])
sys.path.append(chatbotPath)

from chatbot import chatbot


logger = logging.getLogger(__name__)


class chatbotManager:
    bot = None

    @staticmethod
    def initBot():
        logger.info('Initializing bot...')
        chatbotManager.bot = chatbot.Chatbot()

    @staticmethod
    def callBot():
        if chatbotManager.bot:
            logger.info('Bot called!')
        else:
            logger.error('Error: Bot not initialized!')
