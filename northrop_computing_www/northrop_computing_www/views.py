from django.shortcuts import render
from django.http import HttpResponse
# Create your views here.

def index(request):
    html = "Future Site of Northrop Computing - Index under construction" + '<br><br><img src="../static/northrop_computing_www/northrop.jpg"><br><br><a href="/classify/">Image Classifier</a>'
    return HttpResponse(html)
