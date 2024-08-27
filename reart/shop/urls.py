from django.urls import path
from . import views

app_name = 'shop'

urlpatterns = [
    path('shop_home/', views.shop_home, name='shop_home'),
]