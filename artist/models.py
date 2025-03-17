from django.db import models
from django.contrib.auth.models import User
from donors.models import *
from donors.models import InterestRequest

from category.models import Category
from accounts.models import Artist  # Import Artist from accounts app

from django.utils.text import slugify
from django.urls import reverse

# Create your models here.
class Notification(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    message = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    is_read = models.BooleanField(default=False)
    donation=models.ForeignKey('donors.Donation', on_delete=models.CASCADE, null=True,blank=True)

    def __str__(self):
        return f'Notification for {self.user.username}'
    
class Interest(models.Model):
    artist=models.ForeignKey(Artist,on_delete=models.CASCADE)
    donation=models.ForeignKey(Donation, on_delete=models.CASCADE)
    expressed_at=models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f' {self.artist.user.username} - {self.donation}'

class InterestNotification(models.Model):
    artist = models.ForeignKey(Artist, on_delete=models.CASCADE)
    message = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    is_read = models.BooleanField(default=False)
    interest_request=models.ForeignKey(InterestRequest, on_delete=models.CASCADE, null=True,blank=True)

    def __str__(self):
        return f'Notification for {self.artist.user.username}'


class Payment(models.Model):
    artist = models.ForeignKey(Artist, on_delete=models.CASCADE)
    amount = models.DecimalField(max_digits=10, decimal_places=2)
    payment_id = models.CharField(max_length=100, blank=True, null=True)
    order_id = models.CharField(max_length=100, blank=True, null=True)
    status = models.CharField(max_length=20, default='pending')
    created_at = models.DateTimeField(auto_now_add=True)
    interest_request = models.ForeignKey(InterestRequest, on_delete=models.CASCADE, null=True, blank=True)

    def __str__(self):
        return f'Payment {self.id} - {self.artist.user.username}'

class Product(models.Model):
    artist = models.ForeignKey('accounts.Artist', on_delete=models.CASCADE)
    slug = models.SlugField(max_length=255, unique=True, null=True, blank=True)
    name = models.CharField(max_length=255)
    description = models.TextField()
    price = models.DecimalField(max_digits=10, decimal_places=2)
    image = models.ImageField(upload_to='picture/artist/products')
    stock = models.PositiveIntegerField(null=True)
    is_available=models.BooleanField(default=True)
    categories = models.ManyToManyField(Category, blank=True)

    def save(self, *args, **kwargs):
        if not self.slug:
            self.slug = slugify(self.name)
        super(Product, self).save(*args, **kwargs)

    def get_url(self):
        return reverse('product_detail', args=[self.categories.first().slug, self.slug])

    def __str__(self):
        return self.name
    
    def get_completed_purchases(self):
        return Payment.objects.filter(
            status='completed',
            interest_request_donation_medium_of_waste=self.categories.first()
        )

    @classmethod
    def get_all_completed_purchases(cls, artist):
        return Payment.objects.filter(
            status='completed',
            artist=artist
        )
from shop.models import Order
class ProNotification(models.Model):
    artist = models.ForeignKey(Artist, on_delete=models.CASCADE)
    order = models.ForeignKey(Order,on_delete=models.CASCADE, null=True,blank=True)
    message =models.TextField()
    is_read = models.BooleanField(default=False)
    created_at=models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Notification for {self.artist.user.username}"

class ArtistReviewForDonation(models.Model):
    RATING_CHOICES = [
        (1, '1 - Poor'),
        (2, '2 - Fair'),
        (3, '3 - Good'),
        (4, '4 - Very Good'),
        (5, '5 - Excellent'),
    ]
    
    artist = models.ForeignKey(Artist, on_delete=models.CASCADE, related_name='reviews_given')
    donation = models.ForeignKey(Donation, on_delete=models.CASCADE, related_name='reviews')
    donor = models.ForeignKey(Donors, on_delete=models.CASCADE, related_name='donation_reviews')
    interest_request = models.ForeignKey(InterestRequest, on_delete=models.CASCADE, related_name='reviews')
    
    rating = models.IntegerField(choices=RATING_CHOICES)
    title = models.CharField(max_length=255)
    content = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-created_at']
        # Ensure an artist can only review a donation once
        unique_together = ['artist', 'donation']
    
    def __str__(self):
        return f"Review by {self.artist.user.username} for donation {self.donation.id}" 