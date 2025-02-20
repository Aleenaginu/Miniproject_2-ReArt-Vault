from django.db import models
from django.conf import settings
import numpy as np
import json

# Create your models here.

class FaceEncoding(models.Model):
    user = models.OneToOneField(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    encoding = models.TextField()  # Store face encoding as JSON string
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def set_encoding(self, encoding_array):
        """Convert numpy array to JSON string for storage"""
        self.encoding = json.dumps(encoding_array.tolist())

    def get_encoding(self):
        """Convert stored JSON string back to numpy array"""
        return np.array(json.loads(self.encoding))

    def __str__(self):
        return f"Face encoding for {self.user.username}"
