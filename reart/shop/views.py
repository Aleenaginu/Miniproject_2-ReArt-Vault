from django.shortcuts import get_object_or_404, render, redirect
from django.contrib.auth.decorators import login_required
from django.views.decorators.cache import never_cache
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login, logout
from django.contrib import messages
from django.db import transaction


from category.models import Category
from .models import Customers
from artist.models import Product

# Create your views here.

def shop_index(request, category_slug=None):
    if category_slug:
        category=get_object_or_404(Category,slug=category_slug)
        products=Product.objects.filter(categories=category, is_available=True)
    else:
        products=Product.objects.filter(is_available=True)
    products_count=products.count()
    context={
        'products':products,
        'products_count':products_count,
        'category':category if category_slug else None,
    }
    return render(request, 'Customers/index.html',context)

def product_detail(request,category_slug,product_slug):
    category=get_object_or_404(Category,slug=category_slug)
    single_product=get_object_or_404(Product,categories=category,slug=product_slug)
    context={
        'single_product':single_product,
    }
    return render(request, 'Customers/product_detail.html',context)



def customerRegister(request):
    request.session['user_role']='customer'
    if request.method=='POST':
        username=request.POST.get('username')
        
        email=request.POST.get('email')
        phone=request.POST.get('phone')
        profile_pic=request.FILES.get('profile_pic')
        password=request.POST.get('password')
        confirm_password=request.POST.get('confirm_password')
        if password==confirm_password:
            if User.objects.filter(username=username).exists():
                messages.error(request,'Username already exists')
            elif User.objects.filter(email=email).exists():
                messages.error(request,'Email already exists')
            elif Customers.objects.filter(phone=phone).exists():
                messages.error(request,'Phone number already exists')
            else:
                with transaction.atomic():
                    user=User.objects.create_user(username=username,email=email,password=password)
                    Customers.objects.create(user=user,phone=phone,profile_pic=profile_pic)
                messages.success(request,'Registration successfully')
                return redirect('customerlogin')
        else:
            messages.error(request,'Password do not match')
        return redirect('customer_register')
    return render(request, 'Customers/Register.html')


def customerLogin(request):
    if request.method=='POST':
        username=request.POST.get('username')
        password=request.POST.get('password')
        user=authenticate(username=username,password=password)
        if user is not None and hasattr(user,'customers'):
            login(request,user)
            messages.success(request,'Login successfully')
            return redirect('shop_index')
        else:
            messages.error(request,'Invalid username or password')
            return render(request, 'Customers/login.html')
    return render(request, 'Customers/login.html')

def customerLogout(request):
    logout(request)
    request.session.flush()
    messages.success(request,'Logout successfully')
    return redirect('customerlogin')
    


                
     
        
