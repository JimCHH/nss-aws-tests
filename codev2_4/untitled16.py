# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 18:47:19 2021

@author: lab929
"""


import xmltodict

import json
import numpy as np 
import matplotlib.pyplot as plt

with open("1.xml") as xml_file:
    data_dict = xmltodict.parse(xml_file.read())
    json_data = json.dumps(data_dict)
    test = json.loads(json_data)

    xml_file.close()


def convert2str(data):
    
    temp = np.array2string(data, separator = ";")
    string = temp.replace(".", ",")
    string = temp.replace(" ","0")
    string = string[1:]
    string = string[:-1]
    return string


'''
add_str = ""    
for i in range(100):
    add_str += ";-7,97025429501111"
save = []
'''
imp = test["ICSSuiteDBPMRDataSet"]["ICSPatient"]["HITest"][0]["HIImpulse"]
'''

for i in range(22):
    more = imp[i]
    more["EyeVelocitySamples"]  =  more["EyeVelocitySamples"] + add_str
    more["HeadVelocitySamples"]  = add_str + more["HeadVelocitySamples"]     
    
    text = text.replace(",",".").split(";")
    new = []
    for i in text:
        new.append(float(i))
    new = np.array(new)
    save.append(new.shape)
    plt.plot(new)




#test_data1 = convert2str(np.arange(175)/175*500)
#test_data2 = convert2str(np.arange(175)/1000*500 + 1)

#more["EyeVelocitySamples"] = test_data1
#more["HeadVelocitySamples"] = test_data2




counter = 0
for i in imp:
    if i["IsDirectionLeft"] == "true":
        counter +=1
        
        
out = xmltodict.unparse(test, pretty=True)
with open("test2.xml", 'w') as file:
    file.write(out)
'''