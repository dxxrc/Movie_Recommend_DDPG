# Generated by Django 2.2.20 on 2021-04-25 12:47

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('mainsite', '0004_user'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='user',
            name='email',
        ),
    ]