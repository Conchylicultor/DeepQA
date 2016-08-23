from django.shortcuts import render

def mainView(request):
    """ Main view which launch and handle the chatbot view
    Args:
        request (Obj): django request object
    """
    return render(request, 'index.html', {})
