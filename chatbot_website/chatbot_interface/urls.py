from django.conf.urls import url
from . import views

from .chatbotmanager import ChatbotManager

urlpatterns = [
    url(r'^$', views.mainView),
]

# Initialize the chatbot daemon (should be launched only once)
# TODO: Avoid the called when running makemigrations/migrate
ChatbotManager.initBot()
