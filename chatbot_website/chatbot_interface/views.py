from django.shortcuts import render

from .chatbotmanager import ChatbotManager

def mainView(request):
    """ Main view which launch and handle the chatbot
    Args:
        request (Obj): django request object
    """
    ChatbotManager.callBot('')
    return render(request, 'index.html', {})
