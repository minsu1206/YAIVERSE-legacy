from .views import *
from django.urls import path, include

urlpatterns = [
    path('file/', fileView),
    path('file/<str:col>/', fileGetView),
    path('history/<str:user_code>/', historyView),
]
