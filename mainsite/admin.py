from django.contrib import admin
from .models import Movie, User, Ratings

admin.site.register(Movie)
admin.site.register(User)
admin.site.register(Ratings)
