from django.conf.urls import url

from . import views

urlpatterns = [
    url(r'^$', views.index, name='classifier'),
    url(r'^demo$', views.demo, name='demo'),
]
