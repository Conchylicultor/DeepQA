from django.shortcuts import render

from .chatbotmanager import chatbotManager

def mainView(request):
    """ Main view which launch and handle the chatbot
    Args:
        request (Obj): django request object
    """
    chatbotManager.callBot()
    return render(request, 'index.html', {})
