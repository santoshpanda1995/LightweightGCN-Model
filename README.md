# LightweightGCN-Model
A very fast and lightweight model based on Graph Convolutional Network.


<p float="left">
  <img src="/Images/197_low.png" width="200" />
  <img src="/Images/197_high.png" width="200" />
  <img src="/Images/278_low.png" width="200" />
  <img src="/Images/278_high.png" width="200" /> 
</p>

<p float="left">
  <img src="/Images/Low_25.JPG" width="200" />
  <img src="/Images/Ours_25.png" width="200" />
  <img src="/Images/low_722.png" width="200" />
  <img src="/Images/ours_722.png" width="200" /> 
</p>




Here I am giving the steps to be followed in Google colab.
1) To run in colab, first of all clone this repo
```
!git clone https://github.com/santoshpanda1995/LightweightGCN-Model.git
```
The requirements.txt file contains all the library files needed for the program to run.

2) Then copy the contents of requirements.txt and just run in a cell in colab.

**Note:** There may be some extra or repeated library files present in the requirements file, usually I maintain a single requirement file so sometimes i copy the existing command which may already be there. It will never effect anything to the program.

3) We have used pixelshuffle mechanism and the codes for pixelshuffle and inverse pixelshuffle is present in the pixelshuffle.py and inversepixelshuffle.py respectively.

```
!python pixelshuffle.py
!python inversepixelshuffle.py
```

4) After that we have to load our trained model, you can directly download the pretrained model which I have provided and put it in the colab to import it.
```
from tensorflow.keras.models import load_model
# load model
Model = load_model(path to the model)
# summarize model
Model.summary()
```

Now our model is loaded in **Model**. , We can test low light images from this model. Some of the low light images on which i have tested my model , I will provide here.

5) To test image, define this function
```
from google.colab.patches import cv2_imshow
import cv2
def test(img,og):
  height, width, channels = img.shape
  image_for_test= cv.resize(img,(600,400))
  og= cv.resize(og,(600,400))
  image_for_test= image_for_test[np.newaxis,:, :,:]
  Prediction = Model.predict(image_for_test)
  Prediction = Prediction.reshape(400,600,3)
  Prediction = np.array(Prediction)
  #Write the Low light image, the predictiona image and groundtruth image
  cv2.imwrite('root path', img)
  cv2.imwrite('root path', Prediction)
  cv2.imwrite('root path', og)
  original = cv2.imread("/content/low.png")
  compressed = cv2.imread("/content/high.png")
  Hori = np.concatenate((cv.resize(img,(600,400)),cv.resize(og,(600,400)),cv.resize(Prediction,(600,400))), axis=1)
  cv2_imshow(Hori)
  ```
6) Then 
```
img = cv.imread(path of low light image)
og = cv.imread(path of groundtruth image)
test(img,og)
```
