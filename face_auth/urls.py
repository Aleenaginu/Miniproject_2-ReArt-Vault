from django.urls import path
from . import views

urlpatterns = [
    path('verify/', views.verify_face, name='verify_face'),
    path('store/', views.store_face, name='store_face'),
]
