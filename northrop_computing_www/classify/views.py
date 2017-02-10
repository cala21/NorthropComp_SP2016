from django.shortcuts import render
from django.http import HttpResponse
from django.http import HttpResponseRedirect
from .forms import imageForm
# Create your views here.

def index(request):
    if request.method == 'POST':
        form = imageForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            return HttpResponseRedirect('/')
    else:
        form = imageForm()
    return render(request, 'classify/index.html', {
        'form': form
    })

    #return render(request,'classify/index.html',{ })
