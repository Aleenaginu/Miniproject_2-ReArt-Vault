from django.urls import path
from . import views


urlpatterns = [
    path('shop_index/', views.shop_index, name='shop_index'),
    path('customer_register',views.customerRegister, name='customer_register'),
    path('customerlogin', views.customerLogin, name='customerlogin'),
    path('customerlogout', views.customerLogout, name='customerlogout'),
]