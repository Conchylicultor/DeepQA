from django.conf.urls import url
from . import views

from .chatbotmanager import chatbotManager

urlpatterns = [
    url(r'^$', views.mainView),
]

# Initialize the chatbot daemon (should be launched only once)
chatbotManager.initBot()
