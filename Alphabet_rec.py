import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler 
from PIL import Image 
import PIL.ImageOps
import pandas as pd

X=np.load('image.npz')['arr_0']
y=pd.read_csv("labels.csv")["labels"]
classes = ["A",'B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
nclasses=len(classes)

model=LogisticRegression(solver='saga',multi_class='multinomial')

x_train,x_test,y_train,y_test = train_test_split(X,y,random_state=42,train_size=7500,test_size=2500)

sd=StandardScaler()
x_train=sd.fit_transform(x_train)
x_test=sd.transform(x_test)
model.fit(x_train,y_train)
prediction=model.predict(x_test)
accuracy=accuracy_score(y_test,prediction)
print(accuracy)

def get_prediction(img):
    im_pil=Image.open(img)
    image_bw=im_pil.convert('L')
    image_resize=image_bw.resize((22,30),Image.ANTIALIAS)
    image_scaled=sd.transform(np.array(image_resize).reshape(1,660))
    prediction=model.predict(image_scaled)
    
    return prediction