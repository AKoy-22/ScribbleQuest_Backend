# Data models - User, Math_Score and Words_Score tables
# Creating a user will automatically initiate score tables with default score of 0

from django.db import models
from django.core.validators import MinValueValidator, MinLengthValidator, RegexValidator
from django.contrib.auth.models import AbstractUser


class User(AbstractUser):
    grade = models.IntegerField(MinValueValidator(0), null=True, blank=True)
    
    def save(self, *args, **kwargs):
   
        created = not self.pk  # Check if the User is being created for the first time
        super().save(*args, **kwargs)  # Save the User instance

        if created:  # if new user is created, score tables with 0 points at 0 level will be setup
            Maths_Score.objects.create(user=self)
            Words_Score.objects.create(user=self)
    
   

class Maths_Score(models.Model):
    point = models.IntegerField(default=0, validators=[
                                MinValueValidator(0)], verbose_name="Points")
    level = models.IntegerField(default=0, validators=[
                                MinValueValidator(0)], verbose_name="Level")
    user = models.OneToOneField(User, on_delete=models.CASCADE, primary_key=True, related_name='maths_score')

    def __str__(self):
        return f"{self.user} {self.point} {self.level}"


class Words_Score(models.Model):
    point = models.IntegerField(default=0, validators=[
                                MinValueValidator(0)], verbose_name="Points")
    level = models.IntegerField(default=0, validators=[
                                MinValueValidator(0)], verbose_name="Level")
    user = models.OneToOneField(User, on_delete=models.CASCADE, primary_key=True, related_name='words_score')

    def __str__(self):
        return f"{self.user} {self.point} {self.level}"
