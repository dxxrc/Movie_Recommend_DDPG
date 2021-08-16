import random

import numpy as np
import pandas as pd
from django.contrib import messages
from django.http import HttpResponse
from django.shortcuts import render, redirect
from mainsite import models
from mainsite.utils import mypage_genres
from mainsite.utils.mypage import Pagination
from mainsite.models import Movie, User, Ratings


def index(request):
    # # 虚假的腿甲
    # movies = Movie.objects.all()[0:40]

    movies_npy = np.load('./algorithm/test_pred_dict_large_40.npy', allow_pickle = True)
    movies_dict = movies_npy[()]
    movies = Movie.objects.all()[0:40]
    # head是返回的参数字典，前端通过{{ head.name }}调用
    head = {
        'user_photo': '/static/images/user.jpg',
        'login_state': '登录',
        'user_info': '用户未登录',
        'user_name': 'None',
        'user_pwd': 'None',
        'user_gender': 'None',
        'user_age': 'None',
        'user_occupation': 'None',
        'user_phone': 'None',
        'user_email': 'None',
        'user_zip': 'None',
        'movies': movies,
    }
    # 判断登录后返回用户个人信息
    if request.session.get('user_id'):
        user = User.objects.get(id = request.session.get('user_id'))
        head['login_state'] = user.name
        head['user_photo'] = user.photo
        head['user_info'] = '用户信息'
        head['user_name'] = user.name
        head['user_pwd'] = user.pwd
        head['user_gender'] = user.gender
        head['user_age'] = user.age
        head['user_occupation'] = user.occupation
        head['user_phone'] = user.phone_number
        head['user_email'] = user.email
        head['user_zip'] = user.zip
        # 初始化空的列表，存储查询到的电影
        movies = []
        # 把推荐电影集合转换为列表并排序
        movies_list = list(movies_dict[str(user.id)])
        movies_list.sort()
        for i in range(0, 40):
            movie_id = movies_list[i]
            movie = Movie.objects.get(id = movie_id)
            movies.append(movie)
        head['movies'] = movies
    return render(request, 'index.html', {'head': head})


def genres(request):
    # 获取用户选择的类型
    genre = request.GET.get('genre')
    # 在数据库中查找
    movies = Movie.objects.filter(genre__contains = genre)
    # 返回页码
    current_page = request.GET.get('page', 1)
    all_count = movies.count()
    page_obj = mypage_genres.Pagination(current_page = current_page, all_count = all_count, per_page_num = 25,
                                        pager_count = 10, genre = genre)
    page_queryset = movies[page_obj.start:page_obj.end]

    head = {
        'user_photo': '/static/images/user.jpg',
        'login_state': '登录',
        'user_info': '用户未登录',
        'user_name': 'None',
        'user_pwd': 'None',
        'user_gender': 'None',
        'user_age': 'None',
        'user_occupation': 'None',
        'user_phone': 'None',
        'user_email': 'None',
        'user_zip': 'None',
        'movies': movies,
        'genre': genre,
        'page_queryset': page_queryset,
        'page_obj': page_obj,
    }
    # 获取session，更改head
    if request.session.get('user_id'):
        user = User.objects.get(id = request.session.get('user_id'))
        head['login_state'] = user.name
        head['user_photo'] = user.photo
        head['user_info'] = '用户信息'
        head['user_name'] = user.name
        head['user_pwd'] = user.pwd
        head['user_gender'] = user.gender
        head['user_age'] = user.age
        head['user_occupation'] = user.occupation
        head['user_phone'] = user.phone_number
        head['user_email'] = user.email
        head['user_zip'] = user.zip
    return render(request, 'genres.html', {'head': head})


def single(request):
    # 根据基于物品的协同过滤针对电影推荐相似的电影
    path = 'E:\\PyCharm\\Movie_Recommend\\algorithm\\same_genre.csv'
    movie_rec = pd.read_csv(path)

    rec_list = []

    movie_id = request.GET.get('movie_id')
    movie = Movie.objects.get(id = movie_id)

    for i in range(0, 6):
        m = Movie.objects.filter(id = movie_rec[str(i)][int(movie_id)])
        if m:
            rec_list.append(m[0])

    head = {
        'user_photo': '/static/images/user.jpg',
        'login_state': '登录',
        'user_info': '用户未登录',
        'user_name': 'None',
        'user_pwd': 'None',
        'user_gender': 'None',
        'user_age': 'None',
        'user_occupation': 'None',
        'user_phone': 'None',
        'user_email': 'None',
        'user_zip': 'None',
        'movie': movie,
        'rec_list': rec_list,
    }
    if request.session.get('user_id'):
        user = User.objects.get(id = request.session.get('user_id'))
        head['login_state'] = user.name
        head['user_photo'] = user.photo
        head['user_info'] = '用户信息'
        head['user_name'] = user.name
        head['user_pwd'] = user.pwd
        head['user_gender'] = user.gender
        head['user_age'] = user.age
        head['user_occupation'] = user.occupation
        head['user_phone'] = user.phone_number
        head['user_email'] = user.email
        head['user_zip'] = user.zip
    return render(request, 'single.html', {'head': head})


def list_a_z(request):
    # 获取用户选择的电影首字母
    capital = request.GET.get('capital', 'A')
    if capital == 'other':
        movies = Movie.objects.exclude(
            capital__in = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N',
                           'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'])
        current_page = request.GET.get('page', 1)
        all_count = movies.count()
        page_obj = Pagination(current_page = current_page, all_count = all_count, per_page_num = 15,
                              pager_count = 10, capital = capital)
        page_queryset = movies[page_obj.start:page_obj.end]
    else:
        movies = Movie.objects.filter(capital = capital)
        current_page = request.GET.get('page', 1)
        all_count = movies.count()
        page_obj = Pagination(current_page = current_page, all_count = all_count, per_page_num = 15,
                              pager_count = 10, capital = capital)
        page_queryset = movies[page_obj.start:page_obj.end]

    head = {
        'user_photo': '/static/images/user.jpg',
        'login_state': '登录',
        'user_info': '用户未登录',
        'user_name': 'None',
        'user_pwd': 'None',
        'user_gender': 'None',
        'user_age': 'None',
        'user_occupation': 'None',
        'user_phone': 'None',
        'user_email': 'None',
        'user_zip': 'None',
        'movies': movies,
        'all_count': all_count,
        'page_queryset': page_queryset,
        'page_obj': page_obj,
    }
    if request.session.get('user_id'):
        user = User.objects.get(id = request.session.get('user_id'))
        head['login_state'] = user.name
        head['user_photo'] = user.photo
        head['user_info'] = '用户信息'
        head['user_name'] = user.name
        head['user_pwd'] = user.pwd
        head['user_gender'] = user.gender
        head['user_age'] = user.age
        head['user_occupation'] = user.occupation
        head['user_phone'] = user.phone_number
        head['user_email'] = user.email
        head['user_zip'] = user.zip
    return render(request, 'list.html', {'head': head})


def register(request):
    if request.method == 'POST':
        username = request.POST.get("Username")
        password = request.POST.get("Password")
        email = request.POST.get("Email")
        phone = request.POST.get("Phone")
        user = User.objects.filter(name = username)
        if user:
            print('该用户名已存在')
            messages.success(request, '该用户已存在')
        else:
            User.objects.create(
                id = 10000,
                name = username,
                pwd = password,
                email = email,
                phone_number = phone
            )
    return redirect('http://127.0.0.1:8000/')


def login(request):
    username = request.POST.get("Username")
    password = request.POST.get("Password")
    user = User.objects.filter(name = username)
    if user:
        if user[0].pwd == password:
            print('登录成功')
            request.session["user_id"] = user[0].id
        else:
            print('密码错误')
            messages.success(request, '密码错误')
    else:
        print('该用户不存在')
        messages.success(request, '该用户不存在')
    return redirect('http://127.0.0.1:8000/')


def logout(request):
    request.session.flush()
    return redirect('http://127.0.0.1:8000/')


def modify(request):
    user_id = request.session.get('user_id')
    if user_id:
        user_name = request.POST.get("user_name")
        pwd = request.POST.get("pwd")
        user_gender = request.POST.get('user_gender')
        user_age = request.POST.get('user_age')
        user_occupation = request.POST.get('user_occupation')
        user_phone = request.POST.get('user_phone')
        user_email = request.POST.get('user_email')
        user_zip = request.POST.get('user_zip')
        print(user_name + user_gender + user_age + user_occupation + user_phone + user_email + user_zip)
        user = User.objects.get(id = user_id)
        if user_name != '':
            user.name = user_name
        if pwd != '':
            user.pwd = pwd
        if user_gender != '':
            user.gender = user_gender
        if user_age != '':
            user.age = user_age
        if user_occupation != '':
            user.occupation = user_occupation
        if user_phone != '':
            user.phone_number = user_phone
        if user_email != '':
            user.email = user_email
        if user_zip != '':
            user.zip = user_zip
        user.save()
    else:
        messages.success(request, '用户未登录')
    return redirect('http://127.0.0.1:8000/')


# 用户对电影评分
def rating(request):
    if request.session.get('user_id'):
        user_rating = request.POST.get('rating')
        movie_id = request.POST.get('movie_id')
        if isinstance(user_rating, int) and 1 <= user_rating <= 5:
            user = User.objects.get(id = request.session.get('user_id'))
            movie = Movie.objects.get(id = movie_id)
            Ratings.objects.create(
                user_id = user,
                movie_id = movie,
                ratings = user_rating,
            )
        else:
            messages.success(request, '评分必须为1-5的整数')
    else:
        messages.success(request, '用户未登录')

    return redirect('http://127.0.0.1:8000/')


def test(request):
    result = request.GET.get('type')
    messages.success(request, '密码错误')
    print(result)
    return render(request, 'test.html')


# 计算平均分
def import_data_11(request):
    ratings_title = ['UserID', 'MovieID', 'Rating', 'timestamps']
    ratings = pd.read_csv('./algorithm/ml-1m/ratings.dat', sep = '::', header = None, names = ratings_title,
                          engine = 'python')

    ratings_mean = ratings.groupby(['MovieID'], as_index=False).mean()
    for i in range(0, 3706):
        movie = Movie.objects.get(id = ratings_mean['MovieID'][i])
        movie.rating = format(ratings_mean['Rating'][i], '.3f')
        movie.save()
        print(movie)
        print(format(ratings_mean['Rating'][i], '.3f'))

    html = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <title>Title</title>
        </head>
        <body>
            <h1>ok!</h1>
        </body>
        </html>
    """
    return HttpResponse(html)


# 导入用户头像、职业等
def import_data_10(request):
    users_title = ['userID', 'gender', 'age', 'occu', 'zip']
    users_1m = pd.read_csv('./algorithm/ml-1m/users.dat', sep = '::', header = None, names = users_title,
                           engine = 'python')
    users = User.objects.filter().all()
    for i in range(0, 6040):
        print(i)
        for j in range(0, 6040):
            if users[i].id == users_1m['userID'][j]:
                u = users[i]
                print('userID:', u.id)
                # user.photo
                u.photo = '/static/poster/' + str(random.randint(1, 6000)) + '.jpg'
                # user.occupation
                occupation = {'0': 'other or not specified',
                              '1': 'academic/education',
                              '2': 'artist',
                              '3': 'clerical/admin',
                              '4': 'college/grad student',
                              '5': 'customer service',
                              '6': 'doctor/health care',
                              '7': 'executive/managerial',
                              '8': 'farmer',
                              '9': 'homemaker',
                              '10': 'K-12 student',
                              '11': 'lawyer',
                              '12': 'programmer',
                              '13': 'retired',
                              '14': 'sales/marketing',
                              '15': 'scientist',
                              '16': 'self-employed',
                              '17': 'technician/engineer',
                              '18': 'tradesman/craftsman',
                              '19': 'unemployed',
                              '20': 'writer',
                              }
                u.occupation = occupation[str(users_1m['occu'][j])]
                # 性别
                gender = {
                    'F': '女',
                    'M': '男',
                }
                u.gender = gender[str(users_1m['gender'][j])]
                # 邮政编码
                u.zip = users_1m['zip'][j]
                # 年龄
                age = {
                    '1': 'Under 18',
                    '18': '18-24',
                    '25': '25-34',
                    '35': '35-44',
                    '45': '45-49',
                    '50': '50-55',
                    '56': '56+',
                }
                u.age = age[str(users_1m['age'][j])]
                u.save()
        print(u.occupation)

    html = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <title>Title</title>
        </head>
        <body>
            <h1>ok!</h1>
        </body>
        </html>
    """
    return HttpResponse(html)


# 导入评分
def import_data_9(request):
    ratings_title = ['UserID', 'MovieID', 'Rating', 'timestamps']
    ratings = pd.read_csv('./algorithm/ml-1m/ratings.dat', sep = '::', header = None, names = ratings_title,
                          engine = 'python')
    for i in range(0, 1000209):
        user = User.objects.get(id = ratings['UserID'][i])
        movie = Movie.objects.get(id = ratings['MovieID'][i])
        Ratings.objects.create(
            user_id = user,
            movie_id = movie,
            ratings = ratings['Rating'][i]
        )
    html = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <title>Title</title>
        </head>
        <body>
            <h1>ok!</h1>
        </body>
        </html>
    """
    return HttpResponse(html)


# 导入用户
def import_data_8(request):
    ratings_title = ['UserID', 'MovieID', 'Rating', 'timestamps']
    ratings = pd.read_csv('./algorithm/ml-1m/ratings.dat', sep = '::', header = None, names = ratings_title,
                          engine = 'python')
    # user = User.objects.filter().all()

    for i in range(0, len(ratings['UserID'])):
        u = User.objects.filter(id = ratings['UserID'][i])
        if u:
            print('ok')
        else:
            print('8888888888888888888888888888888888888888888888888888888888888888888888888')
            User.objects.create(
                id = ratings['UserID'][i],
                name = 'user_' + str(ratings['UserID'][i]),
                pwd = str(ratings['UserID'][i]),
                phone_number = '15687154399',
                email = '20171300007@mail.ynu.edu.cn',
            )
    html = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <title>Title</title>
        </head>
        <body>
            <h1>ok!</h1>
        </body>
        </html>
    """
    return HttpResponse(html)


# 计算平均分
def import_data_7(request):
    ratings_title = ['UserID', 'MovieID', 'Rating', 'timestamps']
    ratings = pd.read_csv('./algorithm/ml-1m/ratings.dat', sep = '::', header = None, names = ratings_title,
                          engine = 'python')
    movies = Movie.objects.all()
    sum = 0
    times = 0
    a = 0
    for movie in movies:
        for i in range(0, len(ratings["Rating"])):
            if str(ratings['MovieID'][i]).strip() == str(movie.id).strip():
                sum += ratings['Rating'][i]
                times += 1
        a += 1
        if times != 0:
            print('movieid: ' + str(movie.id))
            print(sum / times)
            movie.rating = format(sum / times, '.3f')
            movie.save()

        sum = 0
        times = 0
    html = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <title>Title</title>
        </head>
        <body>
            <h1>ok!</h1>
        </body>
        </html>
    """
    return HttpResponse(html)


# 删除电影
def import_data_6(request):
    movies = Movie.objects.filter().all()
    for u in movies:
        u.delete()
    html = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <title>Title</title>
        </head>
        <body>
            <h1>ok!</h1>
        </body>
        </html>
    """
    return HttpResponse(html)


# 导入电影
def import_data_5(request):
    path_100k = 'E:\\PyCharm\\Movie_Recommend\\algorithm\\IMDBPoster\\info\\info.csv'
    movie_100k = pd.read_csv(path_100k)

    movies_title_1m = ['MovieID', 'Title', 'Genres']
    movie_1m = pd.read_csv('./algorithm/ml-1m/movies.dat', sep = '::', header = None, names = movies_title_1m,
                           engine = 'python')
    print(movie_100k['name'][0])
    print(type(movie_100k['name'][0]))
    print(len(movie_100k['name'][0]))
    print(movie_1m['Title'][0])
    print(type(movie_1m['Title'][0]))
    print(len(movie_1m['Title'][0]))
    if movie_1m['Title'][0].strip() == movie_100k['name'][0].strip():
        print('ok')
    print('movies_1m\n', movie_1m)
    print('movie_100k\n', movie_100k)

    # i for 1m; j for 100k
    # 1m共有3883部电影
    is_insert = False
    for i in range(0, 3883):
        # 100k共有9742部电影
        for j in range(0, 9742):
            if movie_1m['Title'][i].strip() == movie_100k['name'][j].strip():
                is_insert = True
                print(i, ': ', movie_1m['Title'][i].strip())
                print(movie_100k['name'][j].strip())
                m = Movie(
                    id = movie_1m['MovieID'][i],
                    name = movie_1m['Title'][i].strip(),
                    genre = movie_1m['Genres'][i],
                    poster = movie_100k['url'][j],
                    time = movie_100k['time'][j],
                    releasetime = movie_100k['release_time'][j],
                    introduction = movie_100k['intro'][j],
                    directors = movie_100k['directors'][j],
                    writers = movie_100k['writers'][j],
                    actors = movie_100k['starts'][j],
                    capital = movie_1m['Title'][i].strip()[0],
                    rating = 0.000
                )
                m.save()
        if not is_insert:
            print(i, ': ', movie_1m['Title'][i].strip())
            m = Movie(
                id = movie_1m['MovieID'][i],
                name = movie_1m['Title'][i].strip(),
                genre = movie_1m['Genres'][i],
                # poster = '{% static \'images/poster_not_found.png\' %}',
                poster = '/static/images/poster_not_found.png',
                time = '未知',
                releasetime = '未知',
                introduction = '未知',
                directors = '未知',
                writers = '未知',
                actors = '未知',
                capital = movie_1m['Title'][i].strip()[0],
                rating = 0.000
            )
            m.save()
        is_insert = False

    # for i in range(0, 3883):
    #     m = Movie(id = str(movies_1m['MovieID'][i]).strip(), name = str(movies_1m['Title'][i]).strip(),
    #               genre = str(movies_1m['Genres'][i]).strip())
    #     for mov in range(0, 9742):
    #         if str(movie_100k['name'][mov]).strip() == str(m.name).strip():
    #             m.poster = movie_100k['url'][mov]
    #             m.time = movie_100k['time'][mov]
    #             m.releasetime = movie_100k['release_time'][mov]
    #             m.introduction = movie_100k['intro'][mov]
    #             m.directors = movie_100k['directors'][mov]
    #             m.writers = movie_100k['writers'][mov]
    #             m.actors = movie_100k['starts'][mov]
    #             mov_name = str(m.name)
    #             m.capital = mov_name[0]
    #             m.rating = 0.0
    #             m.save()
    #             break

    html = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <title>Title</title>
        </head>
        <body>
            <h1>ok!</h1>
        </body>
        </html>
    """
    return HttpResponse(html)


# 导入用户信息
def import_data_4(request):
    user = User.objects.filter().all()
    for u in user:
        u.delete()
    for i in range(1, 611):
        User.objects.create(
            id = i,
            name = 'user_' + str(i),
            pwd = i,
            phone_number = '15687154399',
            email = '20171300007@mail.ynu.edu.cn',
        )
    html = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <title>Title</title>
        </head>
        <body>
            <h1>ok!</h1>
        </body>
        </html>
    """
    return HttpResponse(html)


# 第一次求电影平均分
def import_data_3(request):
    path = 'E:\\文件\\学习\\数据集\\数据集\\ml-latest-small\\ratings.csv'
    ratings = pd.read_csv(path)
    movies = Movie.objects.all()
    sum = 0
    times = 0
    a = 0
    for movie in movies:
        for rating in range(0, 100836):
            if ratings['movieId'][rating] == movie.id:
                sum += ratings['rating'][rating]
                times += 1
        a += 1
        if times != 0:
            print('movieid: ' + str(movie.id))
            print(sum / times)
            movie.rating = format(sum / times, '.3f')
            movie.save()

        sum = 0
        times = 0

    html = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <title>Title</title>
        </head>
        <body>
            <h1>ok!</h1>
        </body>
        </html>
    """
    return HttpResponse(html)


# 求电影名称首字母
def import_data_2(request):
    movies = Movie.objects.all()

    for mov in movies:
        mov_name = str(mov.name)
        mov.capital = mov_name[0]
        mov.save()

    html = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <title>Title</title>
        </head>
        <body>
            <h1>ok!</h1>
        </body>
        </html>
    """
    return HttpResponse(html)


# 第一次导入电影，100k
def import_data(request):
    path = 'E:\\文件\\学习\\数据集\\电影海报详细信息\\IMDBPoster\\info\\info.csv'
    movie = pd.read_csv(path)
    print(movie['id'][0])

    for i in range(0, 9742):
        m = Movie(id = movie['id'][i], name = movie['name'][i], poster = movie['url'][i], time = movie['time'][i],
                  genre = movie['genre'][i], releasetime = movie['release_time'][i], introduction = movie['intro'][i],
                  directors = movie['directors'][i], writers = movie['writers'][i], actors = movie['starts'][i])
        m.save()

    print('ok')

    html = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <title>Title</title>
        </head>
        <body>
            <h1>ok!</h1>
        </body>
        </html>
    """
    return HttpResponse(html)


# 删除数据库中所有电影信息
def import_data_1(request):
    movie = Movie.objects.all()
    for m in movie:
        m.delete()

    html = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <title>Title</title>
        </head>
        <body>
            <h1>ok!</h1>
        </body>
        </html>
    """
    return HttpResponse(html)
