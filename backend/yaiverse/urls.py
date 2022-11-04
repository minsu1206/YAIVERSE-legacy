from .views import *
from django.urls import path, include

urlpatterns = [
    path('', uploadView),
    path('list', historyAllView),
    path('file/', fileView),
    path('result/<str:col>/', showResultView),
    path('file/<str:col>/', fileGetView),
    path('history/<str:user_code>/', historyView),
]
