from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.views.decorators.cache import never_cache
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login, logout
from django.contrib import messages
from django.db import transaction
from .models import Customers

# Create your views here.

def shop_index(request):
    return render(request, 'Customers/index.html')



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
    


                
     
        
