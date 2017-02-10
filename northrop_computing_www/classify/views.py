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
            return HttpResponseRedirect('/classify/demo')
    else:
        form = imageForm()
    return render(request, 'classify/index.html', {
        'form': form
    })

def demo(request):
    image_data = imageUpload.objects.all().latest('uploaded_at').image#latest('uploaded_at')
    #print image_data
    #html = '<img src="/media/documents/krHJs8F.jpg">'
    return HttpResponse(image_data, content_type="image/jpg")

    #return render(request,'classify/index.html',{ })
