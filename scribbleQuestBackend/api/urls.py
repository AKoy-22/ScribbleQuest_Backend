from django.urls import path

from .import views

urlpatterns = [
    path('', views.getData),
    path('add-user/', views.addUser),
    path('login/', views.loginAuth),
    path('process-image/<slug:imgslug>', views.processImage),
    path('get-random-word/<slug:grade>',views.getRandomWord )
]