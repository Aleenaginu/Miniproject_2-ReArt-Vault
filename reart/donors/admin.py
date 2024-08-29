from django.contrib import admin
from .models import *
# Register your models here.
admin.site.register(Donation)
admin.site.register(DonationImage)
admin.site.register(InterestRequest)
admin.site.register(DonorNotification)