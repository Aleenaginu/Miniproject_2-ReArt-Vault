# Generated by Django 5.0.7 on 2024-09-01 16:13

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('artist', '0006_product_is_available_product_slug'),
    ]

    operations = [
        migrations.DeleteModel(
            name='Product',
        ),
    ]
