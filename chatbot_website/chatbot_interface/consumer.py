from channels import Group
from channels.sessions import channel_session
import logging
import sys
import json

from .chatbotmanager import ChatbotManager


logger = logging.getLogger(__name__)


def _getClientName(client):
    """ Return the unique id for the client
    Args:
        client list<>: the client which send the message of the from [ip (str), port (int)]
    Return:
        str: the id associated with the client
    """
    return 'room-' + client[0] + '-' + str(client[1])


@channel_session
def ws_connect(message):
    """ Called when a client try to open a WebSocket
    Args:
        message (Obj): object containing the client query
    """
    if message['path'] == '/chat':  # Check we are on the right channel
        clientName = _getClientName(message['client'])
        logger.info('New client connected: {}'.format(clientName))
        Group(clientName).add(message.reply_channel)  # Answer back to the client
        message.channel_session['room'] = clientName
        message.reply_channel.send({'accept': True})


@channel_session
def ws_receive(message):
    """ Called when a client send a message
    Args:
        message (Obj): object containing the client query
    """
    # Get client info
    clientName = message.channel_session['room']
    data = json.loads(message['text'])

    # Compute the prediction
    question = data['message']
    try:
        answer = ChatbotManager.callBot(question)
    except:  # Catching all possible mistakes
        logger.error('{}: Error with this question {}'.format(clientName, question))
        logger.error("Unexpected error:", sys.exc_info()[0])
        answer = 'Error: Internal problem'

    # Check eventual error
    if not answer:
        answer = 'Error: Try a shorter sentence'

    logger.info('{}: {} -> {}'.format(clientName, question, answer))

    # Send the prediction back
    Group(clientName).send({'text': json.dumps({'message': answer})})

@channel_session
def ws_disconnect(message):
    """ Called when a client disconnect
    Args:
        message (Obj): object containing the client query
    """
    clientName = message.channel_session['room']
    logger.info('Client disconnected: {}'.format(clientName))
    Group(clientName).discard(message.reply_channel)
