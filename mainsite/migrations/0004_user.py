# Generated by Django 2.2.20 on 2021-04-25 12:38

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('mainsite', '0003_movie_rating'),
    ]

    operations = [
        migrations.CreateModel(
            name='User',
            fields=[
                ('id', models.IntegerField(primary_key=True, serialize=False)),
                ('name', models.CharField(max_length=32)),
                ('pwd', models.CharField(max_length=32)),
                ('phone_number', models.CharField(max_length=32)),
                ('email', models.EmailField(max_length=32)),
            ],
        ),
    ]
