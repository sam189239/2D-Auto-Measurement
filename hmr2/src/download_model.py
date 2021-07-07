## Downloading Model ##
import urllib.request
url = 'https://github.com/russoale/hmr2.0/releases/download/2.0/base_model.paired.zip'
filename = '..\\..\\logs\\paired\\base_model.paired.zip'
urllib.request.urlretrieve(url, filename)
from zipfile import ZipFile
with ZipFile('..\\..\\logs\\paired\\base_model.paired.zip', 'r') as zipObj:
   zipObj.extractall('..\\..\\logs\\paired\\')