# Generated by Django 3.1.7 on 2021-05-05 08:06

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('mainsite', '0012_auto_20210505_1604'),
    ]

    operations = [
        migrations.AddField(
            model_name='user',
            name='zip',
            field=models.CharField(default='000000', max_length=10),
            preserve_default=False,
        ),
    ]