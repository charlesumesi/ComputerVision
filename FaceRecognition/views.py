from django.shortcuts import render
from .models import *
from django.http.response import StreamingHttpResponse

from .camera import *

import requests

"""For implementation in a Django project"""

# Create your views here.


def index(request):
    """View for home.html"""
    return render(request, 'xxx/index.html')


def webgen(cam):
    while True:
        frame = cam.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: frame/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

def webcamfeed(request):
        return StreamingHttpResponse(webgen(Webcam()), 
                                     content_type ='multipart/x-mixed-replace; boundary=frame')
    
def ipwebcamfeed(request):
    if request.COOKIES["url_cookie_"]:
        received_cookie = request.COOKIES["url_cookie_"]
        processed_cookie = received_cookie.split('start_')
        for key in dict_ipwebcamfeed.keys():
            if key == processed_cookie[1]:
                return StreamingHttpResponse(webgen(IPWebcam()), 
                                             content_type ='multipart/x-mixed-replace; boundary=frame')






