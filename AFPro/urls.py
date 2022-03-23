from django.contrib import admin
from django.contrib.auth import views as auth_views
from django.urls import path, include
from users import views as user_views
from django.conf import settings
from django.conf.urls.static import static
from CFD_ML import views as ml_views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('register/', user_views.register, name='register'),
    path('profile/', user_views.profile, name='profile'),
    path('login/', auth_views.LoginView.as_view(template_name='users/login.html'), name='login'),
    path('logout/', auth_views.LogoutView.as_view(template_name='users/logout.html'), name='logout'),
    path('', include('AI_Fraud_Detection.urls')),
    path('classify/', ml_views.classify, name='classify'),
    path('classify/prediction', ml_views.prediction, name='Predict'),
    path('lookup/', ml_views.lookup, name='lookup'),
    path('lookup/singlelookup', ml_views.singlelookup, name='slookup'),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)


