from django.contrib import admin

# Register your models here.
from .models import User, Maths_Score, Words_Score
# Register your models here.

#extends ModelAdmin class

admin.site.register(User)
admin.site.register(Maths_Score)
admin.site.register(Words_Score)
