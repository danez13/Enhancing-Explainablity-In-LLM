# Importing necessary libraries
import os
import urllib.request
from zipfile import ZipFile

# Attempt to create a directory called "glove". If it already exists, do nothing.
try:
    os.mkdir("glove")  # Create directory "glove" to store GloVe data
except:
    pass  # If the directory already exists, ignore the error

# Download the GloVe 6B dataset (pre-trained word vectors) from Stanford's NLP website.
# The dataset is a ZIP file, and we save it as "glove.6B.zip" in the "glove" directory.
urllib.request.urlretrieve("https://nlp.stanford.edu/data/glove.6B.zip", "glove/glove.6B.zip")

# Open the downloaded ZIP file and extract its contents to the "glove" directory.
with ZipFile("glove/glove.6B.zip", 'r') as zObject: 
    zObject.extractall(path="glove")  # Extract all files from the ZIP archive

# Once the extraction is complete, remove the ZIP file to clean up.
os.remove("glove/glove.6B.zip")
