from django.shortcuts import render
from django.http import HttpResponse
from django.http import HttpResponseRedirect
from classify.models import imageUpload
from .forms import imageForm
# Create your views here.

def index(request):
    if request.method == 'POST':
        form = imageForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            image_data = imageUpload.objects.all().latest('uploaded_at')
            return render(request, 'classify/index.html', {
                'form': form,
                'image_data' : image_data,
            })
            #return HttpResponseRedirect('/classify/demo')
    else:
        form = imageForm()
    return render(request, 'classify/index.html', {
        'form': form
    })

def demo(request):
    image_data = imageUpload.objects.all().latest('uploaded_at')
    return render(request, 'classify/demo.html', {
        'image_data': image_data
    })
    #return HttpResponse(image_data, content_type="image/jpg")

    #return render(request,'classify/index.html',{ })
