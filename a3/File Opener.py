import pickle
import zipfile
from PIL import Image

with zipfile.ZipFile('C:\\Users\\evely\\Temp\\IntSys-Education\\a3\\data\\data.zip', 'r') as zip_ref:
    zip_ref.extractall()

pkl_file = open('C:\\Users\\evely\\Temp\\IntSys-Education\\a3\\images.pkl', 'rb')
data = pickle.load(pkl_file)
"""size_dict = {}
for i in range(0, len(data)):
    if size_dict.keys().__contains__(data[i].size):
        size_dict[data[i].size] = size_dict[data[i].size] + 1
    else:
        size_dict[data[i].size] = 1"""

"""for i in range(0, len(data)):
    if(data[i].resize((56, 56)).getbbox()[0]>0 and data[i].resize((56, 56)).getbbox()[1]>0):
        print((i, data[i].getbbox()))
"""

data[10].rotate(45).show()

"""Expected Image Dimensions are 56x56 px"""
"""All images are zoomed in or out- no rectangles to crop."""

pkl_file.close()