# Generated by Django 3.1.7 on 2021-05-03 07:03

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('mainsite', '0008_ratings'),
    ]

    operations = [
        migrations.AddField(
            model_name='user',
            name='Occupation',
            field=models.CharField(default='occupation', max_length=32),
            preserve_default=False,
        ),
    ]
