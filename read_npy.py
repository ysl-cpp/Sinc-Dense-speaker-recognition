import numpy as np
test=np.load('./data_lists/TIMIT_labels.npy',encoding = "latin1")  #加载文件
doc = open('1.txt', 'a')  #打开一个存储文件，并依次写入
print(test, file=doc)  #将打印内容写入文件中