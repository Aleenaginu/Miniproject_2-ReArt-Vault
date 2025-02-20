from django.db import models
from django.contrib.auth.models import User


# Create your models here.

class Customers(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    phone = models.IntegerField(default=9999999999)
    profile_pic = models.ImageField(upload_to='picture/customer', null =True, default='picture/customer/hi.jpg')
    face_encoding = models.BinaryField(null=True, blank=True)
    face_registered = models.BooleanField(default=False)
    
    def __str__(self):
        return self.user.username
    
# from artist.models import Product 
# class Order(models.Model):
#     customer = models.ForeignKey(User, on_delete=models.CASCADE)
#     total_amount = models.DecimalField(max_digits=10, decimal_places=2, default=0)
#     status = models.CharField(max_length=20, default='Pending')
#     created_at = models.DateTimeField(auto_now_add=True)
#     razorpay_order_id = models.CharField(max_length=100, blank=True, null=True)

#     def _str_(self):
#         return f"Order {self.id}"

 
#     def update_total_amount(self):
    
#         total = sum(item.quantity * item.price for item in self.orderitem_set.all())
#         self.total_amount = total
#         self.save()

from artist.models import Product 
class Order(models.Model):
    STATUS_CHOICES = [
        ('Pending', 'Pending'),
        ('Processing', 'Processing'),
        ('Shipped', 'Shipped'),
        ('Delivered', 'Delivered'),
        ('Cancelled', 'Cancelled'),
    ]

    customer = models.ForeignKey(User, on_delete=models.CASCADE)
    total_amount = models.DecimalField(max_digits=10, decimal_places=2, default=0)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='Pending')
    created_at = models.DateTimeField(auto_now_add=True)
    razorpay_order_id = models.CharField(max_length=100, blank=True, null=True)

    def __str__(self):
        return f"Order {self.id}"

    def update_total_amount(self):
        total = sum(item.quantity * item.price for item in self.orderitem_set.all())
        self.total_amount = total
        self.save()

class OrderItem(models.Model):
    order = models.ForeignKey(Order, on_delete=models.CASCADE)
    product = models.ForeignKey(Product, on_delete=models.CASCADE)
    quantity = models.PositiveIntegerField()
    price = models.DecimalField(max_digits=10, decimal_places=2)

    def get_subtotal(self):
        return self.quantity * self.price


class ShippingAddress(models.Model):
    customer = models.ForeignKey(User, on_delete=models.CASCADE)
    order = models.ForeignKey(Order, on_delete=models.CASCADE, related_name='shipping_addresses', null=True, blank=True)
    address_type = models.CharField(max_length=50, blank=True, null=True)
    full_name = models.CharField(max_length=100, blank=True, null=True)
    phone = models.CharField(max_length=15, blank=True, null=True)
    address = models.CharField(max_length=200, blank=True, null=True)
    city = models.CharField(max_length=100, blank=True, null=True)
    state = models.CharField(max_length=100, blank=True, null=True)
    zip_code = models.CharField(max_length=20, blank=True, null=True)
    is_default = models.BooleanField(default=False)
    is_saved = models.BooleanField(default=True)

    def __str__(self):
        return f"{self.address_type} - {self.full_name}"



class Payment(models.Model):
    customer = models.ForeignKey(User, on_delete=models.CASCADE)
    order = models.OneToOneField(Order, on_delete=models.CASCADE)
    amount = models.DecimalField(max_digits=10, decimal_places=2)
    payment_id = models.CharField(max_length=100, blank=True, null=True)
    status = models.CharField(max_length=20, default='Pending')
    shipping_address = models.ForeignKey(ShippingAddress, on_delete=models.SET_NULL, null=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Payment for Order {self.order.id}"
    
class Wishlist(models.Model):
    user = models.ForeignKey(Customers, on_delete=models.CASCADE)
    product = models.ForeignKey(Product, on_delete=models.CASCADE)
    added_at = models.DateTimeField(auto_now_add=True)

class SavedAddress(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    address_type = models.CharField(max_length=50)
    full_name = models.CharField(max_length=100)
    phone = models.CharField(max_length=15)
    address = models.CharField(max_length=200)
    city = models.CharField(max_length=100)
    state = models.CharField(max_length=100)
    zip_code = models.CharField(max_length=20)

    def __str__(self):
        return f"{self.address_type} - {self.full_name}"
