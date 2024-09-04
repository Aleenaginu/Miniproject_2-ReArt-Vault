from django.shortcuts import render,redirect
from artist.models import Product
from .models import Cart,CartItem
from django.contrib.auth.decorators import login_required

# Create your views here.
def _cart_id(request):
    cart=request.session.session_key
    if not cart:
        cart=request.session.create()
    return cart

@login_required(login_url='customerlogin')
def add_cart(request,product_id):
    product=Product.objects.get(id=product_id)
    try:
        cart=Cart.objects.get(cart_id=_cart_id(request))
    except Cart.DoesNotExist:
        cart=Cart.objects.create(
            cart_id=_cart_id(request)
        )
    cart.save()
    try:
        cartitem=CartItem.objects.get(product=product,cart=cart)
        cartitem.Quantity+=1
        cartitem.save()
    except CartItem.DoesNotExist:
        cartitem=CartItem.objects.create(
            product=product,
            cart=cart,
            Quantity=1
        )
        cartitem.save()
    return redirect('cart')

def cart(request,total=0,Quantity=0,cartitems=None):
    try:
        cart=Cart.objects.get(cart_id=_cart_id(request))
        cartitems=CartItem.objects.filter(cart=cart,is_active=True)
        for item in cartitems:
            total+=(item.product.price * item.Quantity)
            Quantity+=item.Quantity
        tax=(2*total)/100
        grand_total=total+tax
    except:
        pass
    context={
        'total':total,
        'Quantity':Quantity,
        'cartitems':cartitems,
        'tax':tax,
        'grand_total':grand_total
    }
    return render(request,'Customers/cart.html',context)
            
   
def remove_cart(request,product_id):
    cart=Cart.objects.get(cart_id=_cart_id(request))
    product=Product.objects.get(id=product_id)
    cartitem=CartItem.objects.get(cart=cart,product=product)
    if cartitem.Quantity > 1:
        cartitem.Quantity-=1
        cartitem.save()
    else:
        cartitem.delete()
    return redirect('cart')

def remove_cartitem(request,product_id):
    cart=Cart.objects.get(cart_id=_cart_id(request))
    product=Product.objects.get(id=product_id)
    cartitem=CartItem.objects.get(cart=cart,product=product)
    cartitem.delete()
    return redirect('cart')
        

