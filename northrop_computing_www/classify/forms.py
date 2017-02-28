from django import forms
from .models import imageUpload

class imageForm(forms.ModelForm):
    class Meta:
        model = imageUpload
        fields = ('image',)

'''
class imageForm(forms.Form):
    class Meta:
        model = imageUpload
        fields = ('description', 'image', )
'''
    #docfile = forms.FileField(
        #label='Select a file',
    #)
#from django import forms


#class DocumentForm(forms.ModelForm):
