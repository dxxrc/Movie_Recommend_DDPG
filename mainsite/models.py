from django.db import models


class Movie(models.Model):
    id = models.IntegerField(primary_key = True)
    name = models.CharField(max_length = 100)
    poster = models.CharField(max_length = 200)
    time = models.CharField(max_length = 100)
    genre = models.CharField(max_length = 100)
    releasetime = models.CharField(max_length = 100)
    introduction = models.CharField(max_length = 1000)
    directors = models.CharField(max_length = 100)
    writers = models.CharField(max_length = 100)
    actors = models.CharField(max_length = 100)
    capital = models.CharField(max_length = 1)
    rating = models.FloatField()

    # 在管理界面，以电影名字为显示内容
    def __str__(self):
        return self.name


class User(models.Model):
    id = models.IntegerField(primary_key = True)
    name = models.CharField(max_length = 32)
    pwd = models.CharField(max_length = 32)
    phone_number = models.CharField(max_length = 32)
    email = models.EmailField(max_length = 32)
    photo = models.CharField(max_length = 100)
    occupation = models.CharField(max_length = 32)
    gender = models.CharField(max_length = 20)
    zip = models.CharField(max_length = 10)
    age = models.CharField(max_length = 10)


class Ratings(models.Model):
    id = models.AutoField(primary_key = True, unique = True)
    user_id = models.ForeignKey('User', on_delete = models.CASCADE)
    movie_id = models.ForeignKey('Movie', on_delete = models.CASCADE)
    ratings = models.IntegerField()
