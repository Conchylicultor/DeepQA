#!/bin/bash

cd chatbot_website/

# Only necessary the first time
python3 manage.py makemigrations
python3 manage.py makemigrations chatbot_interface
python3 manage.py migrate

# Launch the server
python3 manage.py runserver
