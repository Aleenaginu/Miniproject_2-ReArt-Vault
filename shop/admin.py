from django.contrib import admin

# from reart.cart.models import Cart, CartItem
# from reart.category.models import Category

# Register your models here.
from .models import *

# admin.site.register(Category)
# admin.site.register(Product)
# admin.site.register(Cart)
# admin.site.register(CartItem)
admin.site.register(Order)
admin.site.register(OrderItem)
admin.site.register(ShippingAddress)
admin.site.register(Payment)
admin.site.register(Review)
