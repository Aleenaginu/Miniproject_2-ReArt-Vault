from django.contrib import admin
from . models import *
# Register your models here.

admin.site.register(Notification)
admin.site.register(Interest)
admin.site.register(InterestNotification)
admin.site.register(Payment)
admin.site.register(Product)


