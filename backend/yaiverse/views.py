from django.http import JsonResponse
from django.shortcuts import render, HttpResponse, HttpResponseRedirect
from django.views.decorators.csrf import csrf_exempt
from rest_framework.response import Response
from rest_framework import status
from .models import InferenceData
import os
import random
import string
from .apps import YaiverseConfig
import requests


style_mapping = {
        "DISNEY": "disney_preserve_color",
        "JOJO": "jojo_preserve_color",
        "JINX": "arcane_jinx_preserve_color",
        "CAITLYN": "arcane_caitlyn_preserve_color",
        "ART":"art",
        # "YASUHO": "jojo_yasuho_preserve_color",
        "SKETCH": "sketch_multi",
    }

def generate_code(number):
    rand_str = ''
    for _ in range(number):
        rand_str += str(random.choice(string.ascii_letters + string.digits))
    return rand_str


def historyView(request, user_code):
    if request.method == 'GET':
        qs = InferenceData.objects.filter(user_code=user_code).filter(fail=False).order_by('-timestamp').values_list('col', flat=True).distinct()
        result = {'data': list(qs)}
        return JsonResponse(result)
    
    
def historyAllView(request):

    qs = InferenceData.objects.all()
    context = {'qs':qs}

    return render(request, 'list.html', context)



# Create your views here.
@csrf_exempt
def fileView(request):
    if request.method == 'POST':
        file =request.FILES.get('file')
        style = request.POST.get('style', "")
        user_code = request.POST.get('id', "")
        
        print(request.POST)
        col = generate_code(10)
        data = InferenceData.objects.create(col=col, style=style, user_code=user_code)
        img_dir = "/home/yai/backend/data"
        try:
            os.makedirs('{}/{}'.format(img_dir, col))
        except:
            print('error!')
        
        with open('{}/{}/image.jpg'.format(img_dir,col), 'wb') as f:
            for chunk in file.chunks():
                f.write(chunk)
        
        try:
            YaiverseConfig.convertModel.generate_face(col, style_mapping[style])
        except:
            data.fail = True
            data.save()
            return HttpResponse(status=500)
        response = HttpResponse(col)
        return response
    else:
        return HttpResponse('error')
    
    
def uploadView(request):
    if request.method == "POST":
        
        file =request.FILES.get('file')
        style = request.POST.get('style', "")
        user_code = request.POST.get('id', "")
        
        print(request.POST)
        col = generate_code(10)
        InferenceData.objects.create(col=col, style=style, user_code=user_code)
        img_dir = "/home/yai/backend/data"
        try:
            os.makedirs('{}/{}'.format(img_dir, col))
        except:
            print('error!')
        
        with open('{}/{}/image.jpg'.format(img_dir,col), 'wb+') as f:
            for chunk in file.chunks():
                f.write(chunk)
        
        YaiverseConfig.convertModel.generate_face(col, style_mapping[style])
        
        return HttpResponseRedirect('result/{}'.format(col))
    return render(request, 'upload.html')

def showResultView(request, col):
    context = {"col":col}
    return render(request, 'result.html', context)


@csrf_exempt
def fileGetView(request, col):
    img_dir = "/home/yai/backend/data"
    file_loc = '{}/{}/result.jpg'.format(img_dir,col)
    content_type = 'image/jpg'
    filename = "result.jpg"
    with open(file_loc, 'rb') as f:
        file_data = f.read()
    response = HttpResponse(file_data, content_type=content_type)
    response['Content-Disposition'] = 'attachment; filename="{}"'.format(filename)
    return response
	