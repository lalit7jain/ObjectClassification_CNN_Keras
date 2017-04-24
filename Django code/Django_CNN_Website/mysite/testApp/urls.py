from django.conf.urls import url
from django.conf import settings
from testApp import views

urlpatterns = [
    url(r'^$', views.HomePageView.as_view()),
    url(r'^about/$', views.AboutPageView.as_view()),
    url(r'^getData/$', views.getDataView.as_view()),
    url(r'^showData/$', views.index),
    url(r'^simulate/$', views.simulateView.as_view()),
    url(r'^simulateData/$', views.index1),
    url(r'^staticData/$', views.index2),
    url(r'^predictData/$', views.index3),
]



# stuff

(r'^static/(?P<path>.*)$', 'django.views.static.serve',{'document_root': settings.MEDIA_ROOT}),