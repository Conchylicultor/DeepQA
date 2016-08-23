from django.conf import settings
import logging
import sys

chatbotPath = "/".join(settings.BASE_DIR.split('/')[:-1])
sys.path.append(chatbotPath)

from chatbot import chatbot


logger = logging.getLogger(__name__)


class ChatbotManager:
    bot = None

    @staticmethod
    def initBot():
        """ Instantiate the chatbot for later use
        Warning: Should be called only once!
        """
        logger.info('Initializing bot...')
        ChatbotManager.bot = chatbot.Chatbot()
        ChatbotManager.bot.main(['--modelTag', 'server', '--test', 'daemon', '--rootDir', chatbotPath])

    @staticmethod
    def callBot(sentence):
        """ Use the previously instantiated bot to predict a response to the given sentence
        Args:
            sentence (str): the question to answer
        Return:
            str: the answer
        """
        if ChatbotManager.bot:
            logger.info('Bot called!')
            return ChatbotManager.bot.daemonPredict(sentence)
        else:
            logger.error('Error: Bot not initialized!')
