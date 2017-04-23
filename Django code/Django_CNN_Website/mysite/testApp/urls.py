from django.conf.urls import url

from testApp import views

urlpatterns = [
    url(r'^$', views.HomePageView.as_view()),
    url(r'^about/$', views.AboutPageView.as_view()),
    url(r'^getData/$', views.getDataView.as_view()),
    url(r'^showData/$', views.index)
]