from django.conf import settings
import logging
import sys

from django.apps import AppConfig
import sys

chatbotPath = "/".join(settings.BASE_DIR.split('/')[:-1])
sys.path.append(chatbotPath)
from chatbot import chatbot


logger = logging.getLogger(__name__)


class ChatbotManager(AppConfig):
    """ Manage a single instance of the chatbot shared over the website
    """
    name = 'chatbot_interface'
    verbose_name = 'Chatbot Interface'

    bot = None

    def ready(self):
        """ Called by Django only once during startup
        """
        # Initialize the chatbot daemon (should be launched only once)
        if not any(x in sys.argv for x in ['makemigrations', 'migrate']):  # HACK: Avoid initialisation while migrate
            ChatbotManager.initBot()

    @staticmethod
    def initBot():
        """ Instantiate the chatbot for later use
        Should be called only once
        """
        if not ChatbotManager.bot:
            logger.info('Initializing bot...')
            ChatbotManager.bot = chatbot.Chatbot()
            ChatbotManager.bot.main(['--modelTag', 'server', '--test', 'daemon', '--rootDir', chatbotPath])
        else:
            logger.info('Bot already initialized.')

    @staticmethod
    def callBot(sentence):
        """ Use the previously instantiated bot to predict a response to the given sentence
        Args:
            sentence (str): the question to answer
        Return:
            str: the answer
        """
        if ChatbotManager.bot:
            return ChatbotManager.bot.daemonPredict(sentence)
        else:
            logger.error('Error: Bot not initialized!')
