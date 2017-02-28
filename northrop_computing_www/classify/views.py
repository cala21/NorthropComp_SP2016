from django.shortcuts import render
from django.http import HttpResponse
from django.http import HttpResponseRedirect
from classify.models import imageUpload
from .forms import imageForm
import base64
from io import BytesIO, StringIO
import sys

import tensorflow as tf
sys.path.append("../src/")

from image_predictor import *
    
predictor = ImagePredictor(configFile="../configuration.txt",data_dir="../Dataset/FinalRawData",labeled_dir="../Dataset/FinalLabeledData")
graph = tf.get_default_graph()



# Create your views here.

def index(request):
    if request.method == 'POST':
        form = imageForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            image_data = imageUpload.objects.all().latest('uploaded_at')
            pilImg = Image.open(image_data.image)

            global graph
            with graph.as_default():
                classified_data, _, _=  predictor.classify_image(pilImg)

            buffer = BytesIO()
            image = classified_data
            image.save(buffer, format="PNG")
            img_str = "data:image/png;base64," + str(base64.b64encode(buffer.getvalue()))[2:-1]
            return render(request, 'classify/index.html', {
                'form': form,
                'image_data' : image_data,
                'classified_data': img_str
            })
            #return HttpResponseRedirect('/classify/demo')
    else:
        form = imageForm()
    return render(request, 'classify/index.html', {
        'form': form
    })

def demo(request):
    image_data = imageUpload.objects.all().latest('uploaded_at')
    pilImg = Image.open(image_data.image.url)
    classified_data, _, _=  predictor.classify_image(pilImg)

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    img_str = "image/png;base64," + base64.b64encode(buffer.getvalue())
    print(img_str)

    return render(request, 'classify/demo.html', {
        'image_data': image_data,
        'classified_data': img_str
    })
    #return HttpResponse(image_data, content_type="image/jpg")

    #return render(request,'classify/index.html',{ })
