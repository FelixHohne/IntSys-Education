import pickle
import zipfile
from PIL import Image



ct_pkl_file = open('C:\\Users\\evely\\IntSys-Education\\a3\\data\\train\\correct_train_labels.pkl', 'rb')
label_data = pickle.load(ct_pkl_file)

ts_pkl_file = open('C:\\Users\\evely\\IntSys-Education\\a3\\data\\train\\train_samples.pkl', 'rb')
samples = pickle.load(ts_pkl_file)

print(len(list(samples[0].crop().getdata())))

"""missing_labels = []


for i in range(0, len(data)):
    if -1 in data[i]:
        missing_labels.append(data[i])

print(missing_labels)

data[0][3] = 3
data[1][1] = 2
data[2][2] = 3
data[3][0] = 5
data[4][0] = 1
data[5][1] = 1
data[6][3] = 2
data[7][3] = 3
data[8][0] = 6
data[9][3] = 2
data[10][1] = 7
data[11][2] = 7
data[12][1] = 3
data[13][2] = 6
data[14][0] = 0"""

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

#data[0] = data[0].rotate(90)
#data[1].rotate(-45).show()
#data[2].rotate(180).show()
#data[3].rotate(-45).show()
#data[4].rotate(-90).show()
#data[5].rotate(-90).show()
#data[6].rotate(-45).show()
#data[7].rotate(90).show()
#data[8].rotate(180).show()
#data[9].rotate(90).show()
#data[10].rotate(315).show()
#data[11].rotate(180).show()
#data[12].rotate(-90).show()
#data[13].rotate(180).show()
#data[14].rotate(180).show()
#data[15].rotate(0).show()

#data[0].save('C:\\Users\\evely\\IntSys-Education\\a3\\data\\test.png')

"""Expected Image Dimensions are 56x56 px"""
"""All images are zoomed in or out- no rectangles to crop."""

ct_pkl_file.close()
ts_pkl_file.close()