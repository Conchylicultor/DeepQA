#!/bin/bash

echo "Don't use this script for now!"
exit 1

cd chatbot_website/

# Only necessary the first time
#python3 manage.py makemigrations
#python3 manage.py makemigrations chatbot_interface
#python3 manage.py migrate

# Launch the server
redis-server &
python3 manage.py runserver

#daphne chatbot_website.asgi:channel_layer --port 8888
#python manage.py runworker
