# Generated by Django 3.1.7 on 2021-05-03 09:59

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('mainsite', '0010_user_gender'),
    ]

    operations = [
        migrations.RenameField(
            model_name='user',
            old_name='Occupation',
            new_name='occupation',
        ),
    ]