from django.urls import path
from . import views



urlpatterns = [
    path('shop_index/', views.shop_index, name='shop_index'),
    path('<slug:category_slug>/', views.shop_index, name='product_by_category'),
    path('<slug:category_slug>/<slug:product_slug>/', views.product_detail, name='product_detail'),
    path('customer_register',views.customerRegister, name='customer_register'),
    path('customerlogin', views.customerLogin, name='customerlogin'),
    path('customerlogout', views.customerLogout, name='customerlogout'),
]