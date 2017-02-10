from django.conf import settings
from django.conf.urls import url
from django.conf.urls.static import static

from . import views

urlpatterns = [
    url(r'^$', views.index, name='classifier'),
    url(r'^demo$', views.demo, name='demo'),
    url(r'^media/', views.demo, name='demo'),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
