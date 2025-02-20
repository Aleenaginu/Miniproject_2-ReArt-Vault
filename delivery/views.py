from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required, user_passes_test
from django.http import JsonResponse
from django.utils import timezone
from datetime import timedelta
from .models import DeliveryPartner, DeliveryOrder, ApprovedDeliveryPartner, DeliveryStatusUpdate
from django.contrib import messages
from django.db.models import Q
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
from django.contrib.auth.hashers import make_password
from django.db import IntegrityError
from shop.tracking_ws import tracking_handler

# Create your views here.

def is_admin(user):
    return user.is_superuser

def is_delivery_partner(user):
    try:
        delivery_partner = DeliveryPartner.objects.get(user=user)
        return ApprovedDeliveryPartner.objects.filter(delivery_partner=delivery_partner).exists()
    except DeliveryPartner.DoesNotExist:
        return False

@login_required
def homed(request):
    if not is_delivery_partner(request.user):
        logout(request)
        messages.error(request, 'Access denied. Please login as an approved delivery partner.')
        return redirect('logind')
    
    delivery_partner = DeliveryPartner.objects.get(user=request.user)
    
    # Get counts for dashboard
    active_deliveries_count = DeliveryOrder.objects.filter(
        delivery_partner=delivery_partner,
        status__in=['ASSIGNED', 'PICKING_UP', 'PICKED_UP', 'IN_TRANSIT']
    ).count()
    
    completed_today_count = DeliveryOrder.objects.filter(
        delivery_partner=delivery_partner,
        status='DELIVERED',
        updated_at__date=timezone.now().date()
    ).count()
    
    available_orders_count = DeliveryOrder.objects.filter(
        delivery_pincode=delivery_partner.pin_code,
        status='AVAILABLE'
    ).count()
    
    context = {
        'active_deliveries_count': active_deliveries_count,
        'completed_today_count': completed_today_count,
        'available_orders_count': available_orders_count
    }
    return render(request, 'delivery/homed.html', context)

@login_required
def home_delivery(request):
    # Get counts for the dashboard
    active_deliveries_count = DeliveryOrder.objects.filter(
        delivery_partner=request.user.deliverypartner,
        status='IN_PROGRESS'
    ).count()

    completed_today_count = DeliveryOrder.objects.filter(
        delivery_partner=request.user.deliverypartner,
        status='COMPLETED',
        completed_at__date=timezone.now().date()
    ).count()

    available_orders_count = DeliveryOrder.objects.filter(
        status='PENDING',
        delivery_pincode=request.user.deliverypartner.pin_code
    ).count()

    context = {
        'active_deliveries_count': active_deliveries_count,
        'completed_today_count': completed_today_count,
        'available_orders_count': available_orders_count,
    }
    return render(request, 'delivery/homed.html', context)

def logind(request):
    if request.method == 'POST':
        email = request.POST.get('email')
        password = request.POST.get('password')
        
        # First authenticate the user
        user = authenticate(username=email, password=password)
        
        if user is not None:
            # Check if this user is an approved delivery partner
            try:
                delivery_partner = DeliveryPartner.objects.get(email=email)
                is_approved = ApprovedDeliveryPartner.objects.filter(delivery_partner=delivery_partner).exists()
                
                if is_approved:
                    login(request, user)
                    messages.success(request, 'Login successful!')
                    return redirect('homed')
                else:
                    messages.error(request, 'Your account is pending approval. Please wait for admin confirmation.')
            except DeliveryPartner.DoesNotExist:
                messages.error(request, 'Invalid login credentials.')
        else:
            messages.error(request, 'Invalid login credentials.')
    
    return render(request, 'delivery/logind.html')

def registerd(request):
    if request.method == 'POST':
        try:
            # Get form data
            first_name = request.POST.get('firstName')
            last_name = request.POST.get('lastName')
            email = request.POST.get('email')
            phone = request.POST.get('phone')
            date_of_birth = request.POST.get('dateOfBirth')
            address = request.POST.get('address')
            pin_code = request.POST.get('pinnumber')
            driving_license = request.POST.get('drivingLicense')
            aadhaar_number = request.POST.get('aadhaarNumber')
            password = request.POST.get('password')

            # Check if user already exists
            if User.objects.filter(username=email).exists():
                messages.error(request, 'An account with this email already exists. Please use a different email or try logging in.')
                return render(request, 'delivery/registerd.html')

            # Check if delivery partner with this aadhaar exists
            if DeliveryPartner.objects.filter(aadhaar_number=aadhaar_number).exists():
                messages.error(request, 'An account with this Aadhaar number already exists.')
                return render(request, 'delivery/registerd.html')

            # Create User instance
            user = User.objects.create(
                username=email,
                email=email,
                password=make_password(password),
                first_name=first_name,
                last_name=last_name
            )

            # Handle file uploads
            aadhaar_front = request.FILES.get('aadhaarImage')
            aadhaar_back = request.FILES.get('aadhaarImage_back')

            # Create DeliveryPartner instance
            delivery_partner = DeliveryPartner.objects.create(
                user=user,
                first_name=first_name,
                last_name=last_name,
                email=email,
                phone=phone,
                date_of_birth=date_of_birth,
                address=address,
                pin_code=pin_code,
                driving_license=driving_license,
                aadhaar_number=aadhaar_number,
                aadhaar_front_image=aadhaar_front,
                aadhaar_back_image=aadhaar_back
            )

            messages.success(request, 'Registration successful! Please wait for admin confirmation.')
            return render(request, 'delivery/registerd.html', {'registration_success': True})

        except IntegrityError as e:
            if 'auth_user.username' in str(e):
                messages.error(request, 'An account with this email already exists. Please use a different email or try logging in.')
            elif 'delivery_deliverypartner.aadhaar_number' in str(e):
                messages.error(request, 'An account with this Aadhaar number already exists.')
            else:
                messages.error(request, 'Registration failed due to a database error. Please try again.')
            return render(request, 'delivery/registerd.html')
            
        except Exception as e:
            messages.error(request, 'Registration failed. Please check your information and try again.')
            return render(request, 'delivery/registerd.html')

    return render(request, 'delivery/registerd.html')

@login_required
@user_passes_test(is_admin)
def manage_delivery_partners(request):
    # Get all delivery partners that are not approved
    pending_partners = DeliveryPartner.objects.exclude(
        id__in=ApprovedDeliveryPartner.objects.values_list('delivery_partner_id', flat=True)
    ).select_related('user')
    
    # Get all approved delivery partners
    approved_partners = DeliveryPartner.objects.filter(
        id__in=ApprovedDeliveryPartner.objects.values_list('delivery_partner_id', flat=True)
    ).select_related('user')
    
    # Transform the data to match template expectations
    pending_list = [{
        'id': partner.id,
        'first_name': partner.user.first_name,
        'last_name': partner.user.last_name,
        'email': partner.user.email,
        'phone': partner.phone,
        'aadhaar_number': partner.aadhaar_number if hasattr(partner, 'aadhaar_number') else ''
    } for partner in pending_partners]
    
    approved_list = [{
        'id': partner.id,
        'first_name': partner.user.first_name,
        'last_name': partner.user.last_name,
        'email': partner.user.email,
        'phone': partner.phone,
        'aadhaar_number': partner.aadhaar_number if hasattr(partner, 'aadhaar_number') else ''
    } for partner in approved_partners]
    
    context = {
        'pending_partners': pending_list,
        'approved_partners': approved_list
    }
    return render(request, 'admin/manage_delivery.html', context)

@login_required
@user_passes_test(is_admin)
def approve_delivery_partner(request, partner_id):
    delivery_partner = get_object_or_404(DeliveryPartner, id=partner_id)
    
    # Create approved delivery partner entry
    ApprovedDeliveryPartner.objects.create(
        delivery_partner=delivery_partner,
        approved_by=request.user
    )
    
    # Send WhatsApp notification
    from .utils import send_whatsapp_message
    
    message = f"""Congratulations! Your delivery partner registration for ReArt Vault has been approved.
    
You can now log in to your account using your registered email: {delivery_partner.email}
    
Thank you for joining ReArt Vault!"""
    
    # Send the message
    success, result = send_whatsapp_message(delivery_partner.phone, message)
    
    if success:
        messages.success(request, f'Delivery Partner {delivery_partner.first_name} {delivery_partner.last_name} has been approved and notified via WhatsApp.')
    else:
        messages.success(request, f'Delivery Partner {delivery_partner.first_name} {delivery_partner.last_name} has been approved. However, WhatsApp notification failed: {result}')
    
    return redirect('manage_delivery_partners')

@login_required
@user_passes_test(is_admin)
def reject_delivery_partner(request, partner_id):
    delivery_partner = get_object_or_404(DeliveryPartner, id=partner_id)
    user = delivery_partner.user
    
    # Delete the delivery partner and associated user
    delivery_partner.delete()
    user.delete()
    
    messages.success(request, f'Delivery Partner registration has been rejected and removed.')
    return redirect('manage_delivery_partners')

# @login_required
# @user_passes_test(is_admin)
# def delivery_partner_details(request, partner_id):
#     partner = get_object_or_404(DeliveryPartner, id=partner_id)
#     context = {
#         'partner': partner
#     }
#     return render(request, 'delivery/delivery_partner_details.html', context)

@login_required
@user_passes_test(is_delivery_partner)
def available_orders(request):
    """View for showing available orders matching delivery partner's pincode"""
    from shop.models import Order, OrderItem, ShippingAddress
    from artist.models import Product
    
    delivery_partner = DeliveryPartner.objects.get(user=request.user)
    
    print(f"Delivery Partner Pincode: {delivery_partner.pin_code}")
    
    # Get all orders in delivery partner's pincode that are available or being reviewed
    available_orders = DeliveryOrder.objects.filter(
        Q(status='AVAILABLE') | 
        (Q(status='REVIEWING') & Q(locked_by=request.user)),
        delivery_pincode=delivery_partner.pin_code
    ).select_related(
        'order',
        'order__customer',
        'delivery_partner',
        'locked_by'
    ).prefetch_related(
        'order__orderitem_set',
        'order__orderitem_set__product',
        'order__orderitem_set__product__artist',
        'order__shipping_addresses'  
    )
    
    # Debug logging
    print("\nDEBUG: Available Orders Details")
    print("=" * 50)
    
    for delivery_order in available_orders:
        print(f"\nDelivery Order #{delivery_order.id}")
        print(f"Status: {delivery_order.status}")
        print(f"Pincode: {delivery_order.delivery_pincode}")
        
        order = delivery_order.order
        if order:
            print(f"\nOrder #{order.id}")
            print(f"Customer: {order.customer.username} ({order.customer.first_name} {order.customer.last_name})")
            print(f"Total Amount: ₹{order.total_amount}")
            
            # Get shipping address
            shipping_address = order.shipping_addresses.first()
            if shipping_address:
                print(f"Shipping Address: {shipping_address.address}, {shipping_address.city}, {shipping_address.state} {shipping_address.zip_code}")
                # Set customer_address for template
                delivery_order.customer_address = f"{shipping_address.address}, {shipping_address.city}, {shipping_address.state} {shipping_address.zip_code}"
            else:
                print("No shipping address found!")
                delivery_order.customer_address = "Address not available"
            
            print("\nOrder Items:")
            for item in order.orderitem_set.all():
                print(f"- Product: {item.product.name}")
                print(f"  Quantity: {item.quantity}")
                print(f"  Price: ₹{item.price}")
                print(f"  Subtotal: ₹{item.get_subtotal()}")
        else:
            print("No order associated with this delivery order!")
            delivery_order.customer_address = "Order not found"
    
    print("=" * 50)
    
    context = {
        'available_orders': available_orders,
        'current_user': request.user.id
    }
    
    # Debug the context data
    print("\nContext Data:")
    for order in available_orders:
        print(f"\nOrder #{order.order.id} Items:")
        items = list(order.order.orderitem_set.all())
        print(f"Number of items: {len(items)}")
        for item in items:
            print(f"- {item.product.name} x {item.quantity} @ ₹{item.price}")
    
    # Check and release any expired locks
    two_minutes_ago = timezone.now() - timedelta(minutes=2)
    expired_orders = available_orders.filter(
        status='REVIEWING',
        lock_timestamp__lt=two_minutes_ago
    )
    for order in expired_orders:
        order.release_lock()

    return render(request, 'delivery/available_orders.html', context)

@login_required
@user_passes_test(is_delivery_partner)
def lock_order(request, order_id):
    """Lock an order for review"""
    order = get_object_or_404(DeliveryOrder, id=order_id)
    
    if order.can_be_locked_by(request.user):
        order.status = 'REVIEWING'
        order.lock_timestamp = timezone.now()
        order.locked_by = request.user
        order.save()
        
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return JsonResponse({
                'status': 'success',
                'message': 'Order locked for review',
                'expires_at': (order.lock_timestamp + timedelta(minutes=2)).isoformat()
            })
        return redirect('available_orders')
    
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        return JsonResponse({
            'status': 'error',
            'message': 'Order cannot be locked at this time'
        }, status=400)
    messages.error(request, 'Order cannot be locked at this time')
    return redirect('available_orders')

@login_required
@user_passes_test(is_delivery_partner)
def accept_order(request, order_id):
    """Accept a locked order"""
    print(f"\nDEBUG: Accepting order #{order_id}")
    print(f"User: {request.user.username}")
    
    order = get_object_or_404(DeliveryOrder, id=order_id)
    print(f"Order Status: {order.status}")
    print(f"Locked By: {order.locked_by}")
    
    try:
        # Verify the order is locked by this user
        print(f"Checking lock: locked_by={order.locked_by}, status={order.status}")
        if order.locked_by != request.user or order.status != 'REVIEWING':
            error_msg = "Order is not locked by you"
            print(f"Lock check failed: {error_msg}")
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return JsonResponse({
                    'status': 'error',
                    'message': error_msg
                }, status=400)
            messages.error(request, error_msg)
            return redirect('available_orders')
        
        # Get delivery partner
        print("Getting delivery partner")
        delivery_partner = DeliveryPartner.objects.get(user=request.user)
        print(f"Found delivery partner: {delivery_partner}")
        
        # Assign the order
        print("Assigning order")
        order.status = 'ASSIGNED'
        order.delivery_partner = delivery_partner
        order.save()
        print("Order assigned successfully")
        
        # Create status update
        print("Creating status update")
        status_update = DeliveryStatusUpdate.objects.create(
            delivery_order=order,
            status='ASSIGNED',
            updated_by=request.user,
            notes='Order accepted by delivery partner'
        )
        print(f"Status update created: {status_update}")
        
        success_msg = 'Order accepted successfully'
        print(success_msg)
        
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return JsonResponse({
                'status': 'success',
                'message': success_msg
            })
        messages.success(request, success_msg)
        return redirect('my_deliveries')
        
    except DeliveryPartner.DoesNotExist:
        error_msg = 'Delivery partner profile not found'
        print(f"Error: {error_msg}")
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return JsonResponse({
                'status': 'error',
                'message': error_msg
            }, status=400)
        messages.error(request, error_msg)
        return redirect('available_orders')
        
    except Exception as e:
        error_msg = f'An error occurred: {str(e)}'
        print(f"Error: {error_msg}")
        print(f"Exception type: {type(e)}")
        import traceback
        print("Traceback:")
        traceback.print_exc()
        
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return JsonResponse({
                'status': 'error',
                'message': error_msg
            }, status=400)
        messages.error(request, error_msg)
        return redirect('available_orders')

@login_required
@user_passes_test(is_delivery_partner)
def release_order(request, order_id):
    """Release a locked order"""
    if request.method == 'POST':
        order = get_object_or_404(DeliveryOrder, id=order_id)
        
        if order.locked_by == request.user and order.status == 'REVIEWING':
            order.release_lock()
            return JsonResponse({
                'status': 'success',
                'message': 'Order released successfully'
            })
        
        return JsonResponse({
            'status': 'error',
            'message': 'Order cannot be released'
        }, status=400)

@login_required
@user_passes_test(is_delivery_partner)
def update_delivery_status(request, order_id):
    """Update delivery status"""
    if request.method == 'POST':
        try:
            import json
            data = json.loads(request.body)
            
            order = get_object_or_404(DeliveryOrder, id=order_id)
            delivery_partner = DeliveryPartner.objects.get(user=request.user)
            
            # Verify this order belongs to the delivery partner
            if order.delivery_partner != delivery_partner:
                return JsonResponse({
                    'success': False,
                    'message': 'Not authorized to update this order'
                }, status=403)
            
            new_status = data.get('status')
            valid_statuses = [status[0] for status in DeliveryOrder.ORDER_STATUS]
            
            if new_status not in valid_statuses:
                return JsonResponse({
                    'success': False,
                    'message': f'Invalid status. Must be one of: {", ".join(valid_statuses)}'
                }, status=400)
            
            # Update order status
            order.status = new_status
            if new_status == 'DELIVERED':
                order.delivery_time = timezone.now().time()
                order.delivery_confirmation = True
            
            order.save()
            
            # Create status update
            DeliveryStatusUpdate.objects.create(
                delivery_order=order,
                status=new_status,
                notes=data.get('notes', ''),
                location=data.get('location', ''),
                updated_by=request.user
            )

            try:
                # Get the order ID from the delivery order
                shop_order_id = order.order.id
                # Notify tracking clients about the status update
                tracking_handler.notify_status_update(shop_order_id, new_status)
            except Exception as e:
                print(f"Error notifying tracking clients: {str(e)}")
            
            # Get the display version of the status
            status_display = dict(DeliveryOrder.ORDER_STATUS).get(new_status, new_status)
            
            # Get the appropriate color for the status
            status_colors = {
                'ASSIGNED': 'warning',
                'PICKING_UP': 'info',
                'PICKED_UP': 'primary',
                'IN_TRANSIT': 'warning',
                'DELIVERED': 'success'
            }
            
            return JsonResponse({
                'success': True,
                'message': 'Status updated successfully',
                'status': new_status,
                'status_display': status_display,
                'status_color': status_colors.get(new_status, 'secondary')
            })
            
        except Exception as e:
            return JsonResponse({
                'success': False,
                'message': str(e)
            }, status=500)
            
    return JsonResponse({
        'success': False,
        'message': 'Invalid request method'
    }, status=405)

@login_required
@user_passes_test(is_delivery_partner)
def my_deliveries(request):
    """View for showing delivery partner's assigned deliveries"""
    delivery_partner = DeliveryPartner.objects.get(user=request.user)
    
    # Get active deliveries and prefetch related data
    active_deliveries = DeliveryOrder.objects.select_related(
        'order', 'order__customer', 'delivery_partner'
    ).prefetch_related(
        'order__orderitem_set',
        'order__orderitem_set__product',
        'order__orderitem_set__product__artist',
        'status_updates'
    ).filter(
        delivery_partner=delivery_partner,
        status__in=['ASSIGNED', 'PICKING_UP', 'PICKED_UP', 'IN_TRANSIT']
    ).order_by('order__created_at')
    
    # Get completed deliveries
    completed_deliveries = DeliveryOrder.objects.select_related(
        'order', 'order__customer', 'delivery_partner'
    ).prefetch_related(
        'order__orderitem_set',
        'order__orderitem_set__product',
        'order__orderitem_set__product__artist',
        'status_updates'
    ).filter(
        delivery_partner=delivery_partner,
        status='DELIVERED'
    ).order_by('-updated_at')[:10]
    
    for delivery in active_deliveries:
        # Get the latest status update
        latest_update = delivery.status_updates.order_by('-timestamp').first()
        if latest_update:
            delivery.latest_status_update = latest_update
            delivery.status_timestamp = latest_update.timestamp
        else:
            delivery.status_timestamp = delivery.created_at
    
    for delivery in completed_deliveries:
        # Get the completion status update
        completion_update = delivery.status_updates.filter(status='DELIVERED').order_by('-timestamp').first()
        if completion_update:
            delivery.completion_timestamp = completion_update.timestamp
        else:
            delivery.completion_timestamp = delivery.updated_at
    
    context = {
        'active_deliveries': active_deliveries,
        'completed_deliveries': completed_deliveries,
        'status_choices': dict(DeliveryOrder.ORDER_STATUS)
    }
    return render(request, 'delivery/my_deliveries.html', context)

@login_required
def available_orders_new(request):
    # Get all pending orders in the delivery partner's pincode
    available_orders = DeliveryOrder.objects.filter(
        Q(status='PENDING') | 
        (Q(status='REVIEWING') & Q(locked_by=request.user.deliverypartner)),
        delivery_pincode=request.user.deliverypartner.pin_code
    ).select_related('customer', 'order').prefetch_related('items')

    context = {
        'available_orders': available_orders
    }
    return render(request, 'delivery/available_orders.html', context)

@login_required
def lock_order_new(request, order_id):
    order = get_object_or_404(DeliveryOrder, id=order_id)
    
    # Check if order is available and in the delivery partner's pincode
    if order.status != 'PENDING' or order.delivery_pincode != request.user.deliverypartner.pin_code:
        messages.error(request, 'This order is not available for review.')
        return redirect('available_orders_new')
    
    # Lock the order
    order.status = 'REVIEWING'
    order.locked_by = request.user.deliverypartner
    order.locked_at = timezone.now()
    order.save()
    
    messages.success(request, 'Order locked for review. You have 2 minutes to accept or release it.')
    return redirect('available_orders_new')

@login_required
def accept_order_new(request, order_id):
    order = get_object_or_404(DeliveryOrder, id=order_id)
    
    # Check if order is locked by this delivery partner
    if order.status != 'REVIEWING' or order.locked_by != request.user.deliverypartner:
        messages.error(request, 'You cannot accept this order.')
        return redirect('available_orders_new')
    
    # Check if lock hasn't expired (2 minutes)
    if timezone.now() - order.locked_at > timedelta(minutes=2):
        order.status = 'PENDING'
        order.locked_by = None
        order.locked_at = None
        order.save()
        messages.error(request, 'Lock expired. Please try again.')
        return redirect('available_orders_new')
    
    # Accept the order
    order.status = 'IN_PROGRESS'
    order.delivery_partner = request.user.deliverypartner
    order.accepted_at = timezone.now()
    order.save()
    
    messages.success(request, 'Order accepted successfully!')
    return redirect('my_deliveries_new')

@login_required
def release_order_new(request, order_id):
    order = get_object_or_404(DeliveryOrder, id=order_id)
    
    # Check if order is locked by this delivery partner
    if order.status != 'REVIEWING' or order.locked_by != request.user.deliverypartner:
        messages.error(request, 'You cannot release this order.')
        return redirect('available_orders_new')
    
    # Release the order
    order.status = 'PENDING'
    order.locked_by = None
    order.locked_at = None
    order.save()
    
    messages.success(request, 'Order released successfully.')
    return redirect('available_orders_new')

@login_required
def my_deliveries_new(request):
    # Get active and completed deliveries
    active_deliveries = DeliveryOrder.objects.filter(
        delivery_partner=request.user.deliverypartner,
        status='IN_PROGRESS'
    ).select_related('customer', 'order').prefetch_related('items')
    
    completed_deliveries = DeliveryOrder.objects.filter(
        delivery_partner=request.user.deliverypartner,
        status='COMPLETED'
    ).select_related('customer', 'order').prefetch_related('items')
    
    context = {
        'active_deliveries': active_deliveries,
        'completed_deliveries': completed_deliveries
    }
    return render(request, 'delivery/my_deliveries.html', context)

@login_required
def logoutd(request):
    logout(request)
    messages.success(request, 'Successfully logged out.')
    return redirect('logind')
