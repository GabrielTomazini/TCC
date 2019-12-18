import cv2
import os
from skimage.feature import hog
from skimage.io import imread, imshow
from skimage.transform import resize
import matplotlib.pyplot as plt

def main():
    CAMINHO01 = 'dataSet/Bmp/Sample00'
    CAMINHO02 = 'dataSet/Bmp/Sample0'
    path = []
    dataSet = []
    train_vector = []
    hogFeatures = []
    test_vector = []
    train = []
    test = []
    dim = (28,64)
    for n in range(1,63):
        if(n < 10):
            train_ids = next(os.walk('dataSet/Bmp/Sample00'+str(n)))[2]
            lenTrain = int(len(train_ids) * 0.9)
            train.append(lenTrain)
            lenTest = int(len(train_ids) * 0.1)
            test.append(lenTest)
            for m in train_ids:
                way = CAMINHO01 + str(n)+ '/' + m
                path.append(way)
        if(n >=10):
            train_ids = next(os.walk('dataSet/Bmp/Sample0'+str(n)))[2]
            lenTrain = int(len(train_ids) * 0.9)
            train.append(lenTrain)
            lenTest = int(len(train_ids) * 0.1)
            test.append(lenTest)
            for m in train_ids:
                way = CAMINHO02 + str(n)+ '/' + m
                path.append(way)
    cont = 0
    for n in path:
        img = cv2.imread(n,0)
        resized_img = cv2.resize(img,dim)
        fd = hog(resized_img, orientations=9, pixels_per_cell=(8, 8),cells_per_block=(2, 2), multichannel=False)
        hogFeatures.append(fd)
        dataSet.append(resized_img)

    print(hogFeatures)
    '''
    for n in dataSet:
        cv2.imshow('',n)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    '''

if __name__ == "__main__":
    main()
