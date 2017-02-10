from django.db import models

class imageUpload(models.Model):
    id = models.AutoField(primary_key=True)
    image = models.FileField(upload_to='images/')
    uploaded_at = models.DateTimeField(auto_now_add=True)
