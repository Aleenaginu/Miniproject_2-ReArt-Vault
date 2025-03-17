import os
import django

# Set up Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'reart.settings')
django.setup()

from django.contrib.auth.models import User
from accounts.models import Donors
from shop.models import Customers

def fix_donors():
    """
    Create donor profiles for users who have customer profiles but no donor profiles.
    """
    print("Starting to fix donor profiles...")
    
    # Get all users
    users = User.objects.all()
    count = 0
    
    for user in users:
        # Check if user has a customer profile but no donor profile
        has_customer = hasattr(user, 'customers')
        
        # Use try/except to check for donor profile since the related_name might not be set up yet
        has_donor = False
        try:
            has_donor = hasattr(user, 'donors')
        except:
            # Check if there's a Donors object for this user
            has_donor = Donors.objects.filter(user=user).exists()
        
        if has_customer and not has_donor:
            try:
                # Create a donor profile for this user
                phone = user.customers.phone if hasattr(user.customers, 'phone') else 99999999
                donor = Donors.objects.create(
                    user=user,
                    phone=phone
                )
                print(f"Created donor profile for user: {user.username}")
                count += 1
            except Exception as e:
                print(f"Error creating donor profile for {user.username}: {str(e)}")
    
    print(f"Fixed {count} users.")

if __name__ == "__main__":
    fix_donors() 