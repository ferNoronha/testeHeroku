import cv2
import numpy as np
import base64
import io
from imageio import imread


class Pallet:
    def __init__(self):
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        self.K = 5
        
    
    def process(self,encode):
        error = []
        img = imread(io.BytesIO(base64.b64decode(encode)))
        #img = cv2.imread(r'D:\Documentos\DiscoC\brincadeira\ocean.jpg')
        cv2_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        #pltimg = cv2.resize(img,(416,416))
        center = self.Kmeans(cv2_img)
        return center, error
        return "teste"
    


    def Kmeans(self, img):
        image = cv2.resize(img, (200,200))
        np_image = image.reshape((-1,3))
        np_image = np.float32(np_image)
        ret,label,center = cv2.kmeans(np_image, self.K,None,self.criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        
        colors, count = np.unique(label.flatten(),return_counts= True)
        
        union = dict(zip(colors,count))
        
        center = np.uint8(center)
        #print(center)
        centers = []
        for key,value in sorted(union.items(), key= lambda x : x[1]):
            centers.append(center[key].tolist())
        
        cent = []
        for i in centers:
            obj = {
                'R':i[2],
                'G':i[1],
                'B':i[0]
            }
            cent.append(obj)
        
        return cent

