from django.contrib import admin
from .models import *  # Import your models

# Register your models here.
admin.site.register(Donors)
admin.site.register(Artist)
admin.site.register(Adminclick)
admin.site.register(ArtistAddress)