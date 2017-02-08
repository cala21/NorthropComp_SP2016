from django.shortcuts import render
from django.http import HttpResponse
# Create your views here.

def index(request):
    return HttpResponse("Future Site of Northrop Computing Web-Based Classifier")
