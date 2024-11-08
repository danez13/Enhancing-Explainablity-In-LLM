import os
import urllib.request
from zipfile import ZipFile
try:
    os.mkdir("glove")
except:
    pass
urllib.request.urlretrieve("https://nlp.stanford.edu/data/glove.6B.zip", "glove/glove.6B.zip")
with ZipFile("glove/glove.6B.zip", 'r') as zObject: 
    zObject.extractall(path="glove")
os.remove("glove/glove.6B.zip")