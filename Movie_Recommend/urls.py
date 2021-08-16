"""Movie_Recommend URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path

from mainsite.views import index, register, list_a_z, test, single, genres, login, logout, modify, import_data_11, \
    rating

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', index),
    path('register/', register),
    path('login/', login),
    path('list/', list_a_z),
    path('test/', test),
    path('single/', single),
    path('genres/', genres),
    path('logout/', logout),
    path('modify/', modify),
    path('rating/', rating),
    # 向数据库中导入数据
    path('data/', import_data_11),
]
