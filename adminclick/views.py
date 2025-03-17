from django.shortcuts import render,redirect, get_object_or_404

from delivery.models import DeliveryPartner
from delivery.views import is_admin
from .models import MediumOfWaste
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.contrib.auth import authenticate, login, logout
from accounts.models import *
from donors.models import *
from django.core.mail import send_mail
from artist.models import Notification
from django.views.decorators.cache import never_cache
from category.models import *
from django.utils.text import slugify
from django.db.models import Count

# Report Generation Imports
import csv
import xlsxwriter
from io import BytesIO
from django.http import HttpResponse, FileResponse
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from datetime import datetime

from django.db import models

# Create your views here.
def UserLoginadmin(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        user = authenticate(username=username, password=password)
        
        if user is not None and hasattr(user, 'adminclick'):
            login(request, user)
            messages.success(request, 'Login successful')
            return redirect('admin_dashboard')
        else:
            messages.error(request, 'Invalid credentials')
            return render(request, 'admin/adminlogin.html')
    return render(request, 'admin/adminlogin.html')
from django.contrib.auth.decorators import login_required, user_passes_test
@login_required
@user_passes_test(is_admin)
def delivery_partner_details(request, partner_id):
    partner = get_object_or_404(DeliveryPartner, id=partner_id)
    context = {
        'partner': partner
    }
    return render(request, 'admin/delivery_partner_details.html', context)

from django.db.models import Count
@login_required
@never_cache

def admin_dashboard(request):
    if request.user.is_authenticated and request.user.adminclick:
        adminclick = request.user.adminclick
        total_medium_of_waste = MediumOfWaste.objects.count()
        total_donations = Donation.objects.count()
        total_artists = Artist.objects.count()

 
        total_accepted_donations = 0
        total_rejected_donations = 0
        total_pending_donations = 0


        donation_status_stats = Donation.objects.values('status').annotate(count=Count('id'))


        if donation_status_stats:
            total_accepted_donations = donation_status_stats.filter(status='accepted').aggregate(count=Count('id'))['count'] or 0
            total_rejected_donations = donation_status_stats.filter(status='rejected').aggregate(count=Count('id'))['count'] or 0
            total_pending_donations = donation_status_stats.filter(status='pending').aggregate(count=Count('id'))['count'] or 0


        donations_accepted_per_artist = (
            InterestRequest.objects.filter(status='accepted')
            .values('artist__user__username')  
            .annotate(count=Count('id'))
        )

        context = {
            'adminclick': adminclick,
            'total_medium_of_waste': total_medium_of_waste,
            'total_donations': total_donations,
            'total_artists': total_artists,
            'donation_status_stats': donation_status_stats,
            'donations_accepted_per_artist': donations_accepted_per_artist,
            'total_accepted_donations': total_accepted_donations,
            'total_rejected_donations': total_rejected_donations,
            'total_pending_donations': total_pending_donations,
        }
        return render(request, 'admin/dashboard.html', context)
@login_required
@never_cache
def approve_artists(request):
    adminclick= request.user.adminclick
    pending_artists = Artist.objects.filter(is_approved=False)
    artists = Artist.objects.all()
    total_medium_of_waste=MediumOfWaste.objects.count()
    total_donations=Donation.objects.count()
    total_artists=Artist.objects.count()
    context={
        'pending_artists': pending_artists,
        'artists': artists,
         'adminclick':adminclick,
            'total_medium_of_waste':total_medium_of_waste,
            'total_donations':total_donations,
            'total_artists':total_artists

    }

    return render(request, 'admin/approve_artists.html', context)

@login_required
@never_cache
def approve_artist(request, artist_id):

    artist = get_object_or_404(Artist, id=artist_id)
    artist.is_approved = True
    artist.save()

    return redirect('approve_artists')


def reject_artist(request, artist_id):

    artist = get_object_or_404(Artist, id=artist_id)
    artist.delete()

    return redirect('approve_artists')

def artist_details(request, artist_id):
    adminclick= request.user.adminclick
    total_medium_of_waste=MediumOfWaste.objects.count()
    total_donations=Donation.objects.count()
    total_artists=Artist.objects.count()
    artist=get_object_or_404(Artist, id=artist_id)
    context={
        'artist': artist,
         'adminclick':adminclick,
            'total_medium_of_waste':total_medium_of_waste,
            'total_donations':total_donations,
            'total_artists':total_artists
    }
    return render(request, 'admin/artist_details.html', context)


@login_required
@never_cache
def add_medium_of_waste(request):
    adminclick= request.user.adminclick
    total_medium_of_waste=MediumOfWaste.objects.count()
    total_donations=Donation.objects.count()
    total_artists=Artist.objects.count()
    context={
         'adminclick':adminclick,
            'total_medium_of_waste':total_medium_of_waste,
            'total_donations':total_donations,
            'total_artists':total_artists
    }

    if request.method == 'POST':
        
        name = request.POST.get('name')
        description = request.POST.get('description')
        rate = request.POST.get('rate')
        
        
        if MediumOfWaste.objects.filter(name=name).exists():
            messages.info(request,"Already Exist")
            return render(request, 'admin/add_medium_of_waste.html')

        
        MediumOfWaste.objects.create(name=name, description=description, rate=rate)
        messages.success(request,"Medium of waste added successfully")
        return redirect('admin_dashboard')  

    return render(request, 'admin/add_medium_of_waste.html', context)



@login_required
@never_cache
def set_rates(request):
    adminclick=request.user.adminclick
    total_medium_of_waste=MediumOfWaste.objects.count()
    total_donations=Donation.objects.count()
    total_artists=Artist.objects.count()
    if request.method=='POST':
        for medium_id, rate in request.POST.items():
            if medium_id.startswith('rate_'):
                try:
                    medium_id=int(medium_id.split('_')[1])
                    medium=MediumOfWaste.objects.get(id=medium_id)
                    medium.rate=float(rate)
                    medium.save()
                except (ValueError, MediumOfWaste.DoesNotExist):
                    continue
        messages.success(request,'Rates set successfully')
        return redirect('set_rates')
    mediums=MediumOfWaste.objects.all()
    context={
        'mediums': mediums,
         'adminclick':adminclick,
            'total_medium_of_waste':total_medium_of_waste,
            'total_donations':total_donations,
            'total_artists':total_artists
    }
    return render(request,'admin/set_rates.html',context)


@login_required
@never_cache
def donation_listview(request):
    adminclick= request.user.adminclick
    total_medium_of_waste=MediumOfWaste.objects.count()
    total_donations=Donation.objects.count()
    total_artists=Artist.objects.count()
    donations = Donation.objects.select_related('donor', 'medium_of_waste').all()
    context={
    'donations': donations,
    'adminclick':adminclick,
    'total_medium_of_waste':total_medium_of_waste,
    'total_donations':total_donations,
    'total_artists':total_artists

    }
    return render(request, 'admin/donation_list.html', context)


# @login_required
# @never_cache
# def donation_detail(request, pk):
#     donation = get_object_or_404(Donation, pk=pk)

#     if request.method == 'POST':
#         status = request.POST.get('status')
#         if status == 'rejected':
#             donation.delete()
#             return redirect('donation_listview')
#         elif status in dict(Donation.STATUS_CHOICES):
#             donation.status = status
#             donation.save()
            
#             if status == 'accepted':
               
#                 artists = Artist.objects.filter(medium=donation.medium_of_waste, is_approved=True)
#                 for artist in artists:
                 
#                     send_mail(
#                         'Donation Accepted',
#                         f'A new waste donation in your medium has been accepted:\n'
#                         f'Medium: {donation.medium_of_waste.name}\n'
#                         f'Quantity: {donation.quantity}\n'
#                         f'Location: {donation.location}\n',
#                         'reartvault@gmail.com',
#                         [artist.user.email],
#                         fail_silently=False,
#                     )
                    
#                     Notification.objects.create(
#                         user=artist.user,
#                         message=f'New waste donation in your medium: {donation.medium_of_waste.name}. Quantity: {donation.quantity}, Location: {donation.location}.',
#                         donation=donation,
#                     )

#             return redirect('donation_listview')

#     return render(request, 'admin/donation_detail.html', {'donation': donation})
@login_required
def donation_detail(request, pk):
    donation = get_object_or_404(Donation, pk=pk)
    adminclick = request.user.adminclick
    context = {
        'donation': donation,
        'adminclick': adminclick
    }
    if request.method == 'POST':
        status = request.POST.get('status')
        if status == 'rejected':
            donation.delete()
            return redirect('donation_listview')
        elif status in dict(Donation.STATUS_CHOICES):
            donation.status = status
            donation.save()

            if status == 'accepted':
             
                artists = Artist.objects.filter(mediums__in=[donation.medium_of_waste], is_approved=True).distinct()
                for artist in artists:
                   
                    send_mail(
                        'Donation Accepted',
                        f'A new waste donation in one of your mediums has been accepted:\n'
                        f'Medium: {donation.medium_of_waste.name}\n'
                        f'Quantity: {donation.quantity}\n'
                        f'Location: {donation.location}\n',
                        'your-email@example.com',
                        [artist.user.email],
                        fail_silently=False,
                    )
                 
                    Notification.objects.create(
                        user=artist.user,
                        message=f'New waste donation in your medium: {donation.medium_of_waste.name}. Quantity: {donation.quantity}, Location: {donation.location}.',
                        donation=donation,
                    )

            return redirect('donation_listview')

    return render(request, 'admin/donation_detail.html', context)


@login_required
def add_category(request):

    if request.method == 'POST':
        category_name=request.POST.get('category_name')

        if category_name:
            slug=slugify(category_name)

            if Category.objects.filter(slug=slug).exists():
                messages.info(request,'Category with this name already exists')
                return redirect('add_category')
            Category.objects.create(name=category_name, slug=slug)
            messages.success(request,'Category added successfully')
            return redirect('add_category')
            
        messages.error(request,'Category name is required')
    return render(request,'admin/add_category.html')

from django.db import models

def admin_search(request):
    query = request.GET.get('query', '')  
    artists = Artist.objects.filter(is_approved=True)
    accepted_donations = Donation.objects.filter(status='accepted')

    if query:
 
        artists = artists.filter(
            models.Q(user__username__icontains=query) |
            models.Q(mediums__name__icontains=query)
        ).distinct()

      
        accepted_donations = accepted_donations.filter(
            medium_of_waste__name__icontains=query  # Corrected field name
        )

    context = {
        'artists': artists,
        'accepted_donations': accepted_donations,
        'query': query  
    }
    return render(request, 'admin/search_results.html', context)




# def artist_list(request):
#     artists = Artist.objects.all()
#     return render(request, 'admin/artist_list.html', {'artists': artists})

@login_required
@user_passes_test(is_admin)
def generate_report(request):
    if request.method == 'POST':
        report_type = request.POST.get('report_type')
        report_format = request.POST.get('report_format')
        
        # Get data based on report type
        data = []
        headers = []
        filename = f"{report_type}_report"
        
        if report_type == 'artist_approved':
            queryset = Artist.objects.filter(is_approved=True)
            headers = ['ID', 'Username', 'Email', 'Date Joined', 'Phone', 'Approved']
            data = [
                [artist.id, artist.user.username, artist.user.email, 
                 artist.user.date_joined.strftime('%Y-%m-%d'), 
                 artist.phone, 'Yes'] 
                for artist in queryset
            ]
            title = "Approved Artists Report"
            
        elif report_type == 'artist_pending':
            queryset = Artist.objects.filter(is_approved=False)
            headers = ['ID', 'Username', 'Email', 'Date Joined', 'Phone', 'Approved']
            data = [
                [artist.id, artist.user.username, artist.user.email, 
                 artist.user.date_joined.strftime('%Y-%m-%d'), 
                 artist.phone, 'No'] 
                for artist in queryset
            ]
            title = "Pending Artists Report"
            
        elif report_type == 'donation_all':
            queryset = Donation.objects.all()
            headers = ['ID', 'Donor', 'Medium', 'Status', 'Date', 'Location', 'Quantity']
            data = [
                [donation.id, donation.donor.user.username, donation.medium_of_waste.name, 
                 donation.status, donation.date_donated.strftime('%Y-%m-%d'), 
                 donation.location, donation.quantity] 
                for donation in queryset
            ]
            title = "All Donations Report"
            
        elif report_type == 'donation_accepted':
            queryset = Donation.objects.filter(status='accepted')
            headers = ['ID', 'Donor', 'Medium', 'Date', 'Location', 'Quantity']
            data = [
                [donation.id, donation.donor.user.username, donation.medium_of_waste.name, 
                 donation.date_donated.strftime('%Y-%m-%d'), 
                 donation.location, donation.quantity] 
                for donation in queryset
            ]
            title = "Accepted Donations Report"
            
        elif report_type == 'donation_rejected':
            queryset = Donation.objects.filter(status='rejected')
            headers = ['ID', 'Donor', 'Medium', 'Date', 'Location', 'Quantity']
            data = [
                [donation.id, donation.donor.user.username, donation.medium_of_waste.name, 
                 donation.date_donated.strftime('%Y-%m-%d'), 
                 donation.location, donation.quantity] 
                for donation in queryset
            ]
            title = "Rejected Donations Report"
            
        elif report_type == 'donation_pending':
            queryset = Donation.objects.filter(status='pending')
            headers = ['ID', 'Donor', 'Medium', 'Date', 'Location', 'Quantity']
            data = [
                [donation.id, donation.donor.user.username, donation.medium_of_waste.name, 
                 donation.date_donated.strftime('%Y-%m-%d'), 
                 donation.location, donation.quantity] 
                for donation in queryset
            ]
            title = "Pending Donations Report"
            
        elif report_type == 'medium_waste':
            queryset = MediumOfWaste.objects.all()
            headers = ['ID', 'Name', 'Description', 'Rate']
            data = [
                [medium.id, medium.name, medium.description, medium.rate] 
                for medium in queryset
            ]
            title = "Medium of Waste Report"
            
        elif report_type == 'delivery_partners':
            queryset = DeliveryPartner.objects.all()
            headers = ['ID', 'Name', 'Contact', 'Vehicle Number', 'Location']
            data = [
                [partner.id, partner.user.username, partner.contact_number, 
                 partner.vehicle_number, partner.location] 
                for partner in queryset
            ]
            title = "Delivery Partners Report"
        
        # Generate report based on format
        if report_format == 'pdf':
            return generate_pdf_report(title, headers, data, filename)
        elif report_format == 'csv':
            return generate_csv_report(headers, data, filename)
        elif report_format == 'excel':
            return generate_excel_report(headers, data, filename)
    
    # If something goes wrong, redirect back to dashboard
    messages.error(request, 'Failed to generate report. Please try again.')
    return redirect('admin_dashboard')


def generate_pdf_report(title, headers, data, filename):
    response = HttpResponse(content_type='application/pdf')
    response['Content-Disposition'] = f'attachment; filename="{filename}.pdf"'
    
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    elements = []
    
    # Add title
    styles = getSampleStyleSheet()
    elements.append(Paragraph(title, styles['Title']))
    elements.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    
    # Add table
    table_data = [headers] + data
    table = Table(table_data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    
    elements.append(table)
    doc.build(elements)
    
    pdf = buffer.getvalue()
    buffer.close()
    response.write(pdf)
    
    return response


def generate_csv_report(headers, data, filename):
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = f'attachment; filename="{filename}.csv"'
    
    writer = csv.writer(response)
    writer.writerow(headers)
    for row in data:
        writer.writerow(row)
    
    return response


def generate_excel_report(headers, data, filename):
    output = BytesIO()
    workbook = xlsxwriter.Workbook(output)
    worksheet = workbook.add_worksheet()
    
    # Define formats
    header_format = workbook.add_format({
        'bold': True,
        'bg_color': '#4B5320',
        'color': 'white',
        'align': 'center',
        'valign': 'vcenter',
        'border': 1
    })
    
    cell_format = workbook.add_format({
        'align': 'left',
        'valign': 'vcenter',
        'border': 1
    })
    
    # Write headers
    for col_num, header in enumerate(headers):
        worksheet.write(0, col_num, header, header_format)
    
    # Write data rows
    for row_num, row_data in enumerate(data):
        for col_num, cell_data in enumerate(row_data):
            worksheet.write(row_num + 1, col_num, cell_data, cell_format)
    
    # Auto-fit columns
    for i, _ in enumerate(headers):
        worksheet.set_column(i, i, 15)
    
    workbook.close()
    
    # Prepare the response
    output.seek(0)
    response = HttpResponse(
        output.read(),
        content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )
    response['Content-Disposition'] = f'attachment; filename="{filename}.xlsx"'
    
    return response
