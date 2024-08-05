# yourapp/pipeline.py
from social_core.exceptions import AuthForbidden
from accounts.models import Donors,Artist


def set_user_role(backend, user, response, *args, **kwargs):
    if backend.name == 'google-oauth2':
        user_role = backend.strategy.session_get('user_role')
        if user and user_role:
            if user_role == 'donor':
                if not Donors.objects.filter(user=user).exists():  # Check if user is already a donor
                    Donors.objects.create(user=user)
            elif user_role == 'artist':
                if not Artist.objects.filter(user=user).exists():  # Check if user is already an artist
                    Artist.objects.create(user=user)
                