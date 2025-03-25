import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import pandas
from PIL import Image

class TotalDataset(Dataset):
    def __init__(self, label, imageSize = 224, aug = False):
        self.label = label
        self.lenght = self.label.shape[0]
        self.aug = aug
        self.imageSize = imageSize
        self.imageTransform = imageTransform = transforms.Compose([
         transforms.Resize((imageSize, imageSize)),
            transforms.ToTensor()
        ])
        
    def __getitem__(self, index):
        pathImage = '/data/cino/Datasets/ISIC/ISIC_2019_Training_Input/' + self.label['image'][index] + '.jpg'
        label = np.argmax(np.array(self.label.loc[index][1:], dtype = 'float32' )[:-1])
        image = self.imageTransform(Image.open(pathImage))
        if self.aug:
            image = randomTransform(image)
        return (image, torch.tensor(label))
    
    def getKFolds(self, k = 5):
        kFolds = []
        
        df = self.label
        
        for index in range(k):
            for label in ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]:
                numClassImagesVal = round(len(df[df[label]==1.]) / k)
                valStart = round(numClassImagesVal * index)
                valEnd = round(numClassImagesVal * (index + 1))

                if label == "MEL":
                    dfTest = df[df[label]==1.].iloc[valStart:valEnd]
                else:
                    dfTest = pandas.concat([dfTest, df[df[label]==1.].iloc[valStart:valEnd]])
                    

            dfTest = dfTest.reset_index(drop=True)
            dfTrain = pandas.concat([df,dfTest]).drop_duplicates(keep=False).reset_index(drop=True)
                
            #Create the dataset classes
            datasetTrain = TotalDataset(dfTrain, imageSize = self.imageSize, aug=True)
            datasetTest = TotalDataset(dfTest, imageSize = self.imageSize)
            
            kFolds.append((datasetTrain, datasetTest))
    
        return kFolds
        
        
    def __len__(self):
        return self.lenght