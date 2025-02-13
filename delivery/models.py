from django.db import models
from django.contrib.auth.models import User

# Create your models here.

class DeliveryPartner(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, null=True, blank=True)
    first_name = models.CharField(max_length=100, null=True, blank=True)
    last_name = models.CharField(max_length=100, null=True, blank=True)
    email = models.EmailField(unique=True, null=True, blank=True)
    phone = models.CharField(max_length=15, null=True, blank=True)
    date_of_birth = models.DateField(null=True, blank=True)
    address = models.TextField(null=True, blank=True)
    pin_code = models.CharField(max_length=6, null=True, blank=True)
    driving_license = models.CharField(max_length=20, null=True, blank=True)
    aadhaar_number = models.CharField(max_length=12, unique=True, null=True, blank=True)
    aadhaar_front_image = models.FileField(upload_to='delivery_partner/aadhaar/', null=True, blank=True)
    aadhaar_back_image = models.FileField(upload_to='delivery_partner/aadhaar/', null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True, null=True, blank=True)
    updated_at = models.DateTimeField(auto_now=True, null=True, blank=True)

    def __str__(self):
        return f"{self.first_name} {self.last_name}"

class ApprovedDeliveryPartner(models.Model):
    delivery_partner = models.OneToOneField(DeliveryPartner, on_delete=models.CASCADE)
    approved_at = models.DateTimeField(auto_now_add=True)
    approved_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, related_name='approved_delivery_partners')

    def __str__(self):
        return f"Approved: {self.delivery_partner.first_name} {self.delivery_partner.last_name}"

class DeliveryOrder(models.Model):
    ORDER_STATUS = [
        ('AVAILABLE', '🟢 Available'),
        ('REVIEWING', '🟡 Being Reviewed'),
        ('ASSIGNED', '🔴 Assigned'),
        ('PICKING_UP', 'Picking Up'),
        ('PICKED_UP', 'Picked Up'),
        ('IN_TRANSIT', 'In Transit'),
        ('DELIVERED', 'Delivered'),
        ('CANCELLED', 'Cancelled')
    ]

    order = models.ForeignKey('shop.Order', on_delete=models.CASCADE, related_name='delivery_orders', null=True, blank=True)
    delivery_partner = models.ForeignKey(DeliveryPartner, on_delete=models.SET_NULL, null=True, blank=True)
    status = models.CharField(max_length=20, choices=ORDER_STATUS, default='AVAILABLE', null=True, blank=True)
    
    # Pincode matching
    delivery_pincode = models.CharField(max_length=6, null=True, blank=True)
    
    # Order locking system
    lock_timestamp = models.DateTimeField(null=True, blank=True)
    locked_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True, related_name='locked_orders')
    
    # Pickup details
    artist_address = models.TextField(null=True, blank=True)
    artist_pincode = models.CharField(max_length=6, null=True, blank=True)
    pickup_date = models.DateField(null=True, blank=True)
    pickup_time = models.TimeField(null=True, blank=True)
    pickup_notes = models.TextField(blank=True, null=True)
    pickup_confirmation = models.BooleanField(default=False)
    
    # Delivery details
    customer_address = models.TextField(null=True, blank=True)
    customer_pincode = models.CharField(max_length=6, null=True, blank=True)
    expected_delivery_date = models.DateField(null=True, blank=True)
    delivery_time = models.TimeField(null=True, blank=True)
    delivery_notes = models.TextField(blank=True, null=True)
    delivery_confirmation = models.BooleanField(default=False)
    customer_signature = models.ImageField(upload_to='signatures/', null=True, blank=True)
    
    # Tracking
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"Delivery #{self.id} - {self.get_status_display()}"

    def get_status_color(self):
        """Return Bootstrap color class based on status"""
        status_colors = {
            'AVAILABLE': 'success',
            'REVIEWING': 'warning',
            'ASSIGNED': 'danger',
            'PICKING_UP': 'info',
            'PICKED_UP': 'primary',
            'IN_TRANSIT': 'info',
            'DELIVERED': 'success',
            'CANCELLED': 'secondary'
        }
        return status_colors.get(self.status, 'secondary')

    def release_lock(self):
        """Release the lock on this order"""
        self.status = 'AVAILABLE'
        self.lock_timestamp = None
        self.locked_by = None
        self.save()

    def can_be_locked_by(self, user):
        """Check if the order can be locked by the given user"""
        if self.status != 'AVAILABLE':
            return False
        
        # Check if user is a delivery partner
        try:
            delivery_partner = DeliveryPartner.objects.get(user=user)
            return delivery_partner.pin_code == self.delivery_pincode
        except DeliveryPartner.DoesNotExist:
            return False

class DeliveryStatusUpdate(models.Model):
    delivery_order = models.ForeignKey(DeliveryOrder, on_delete=models.CASCADE, related_name='status_updates')
    status = models.CharField(max_length=20, choices=DeliveryOrder.ORDER_STATUS)
    notes = models.TextField(blank=True, null=True)
    location = models.CharField(max_length=255, blank=True, null=True)
    timestamp = models.DateTimeField(auto_now_add=True)
    updated_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True)

    def __str__(self):
        return f"Status Update for Delivery #{self.delivery_order.id} - {self.status}"

    class Meta:
        ordering = ['-timestamp']
