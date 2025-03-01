from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('AQI/', views.mainfunction , name='mainfunction')
]