from django.shortcuts import render
from django.http.response import StreamingHttpResponse

from .camera import *


"""For implementation in a Django project"""

# Create your views here.


def index(request):
    """View for index.html"""
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
    return StreamingHttpResponse(webgen(IPWebcam()), 
                                 content_type ='multipart/x-mixed-replace; boundary=frame')






