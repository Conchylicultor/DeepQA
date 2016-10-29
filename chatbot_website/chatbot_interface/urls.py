from django.conf.urls import url
from . import views

from .chatbotmanager import ChatbotManager

urlpatterns = [
    url(r'^$', views.mainView),
]
