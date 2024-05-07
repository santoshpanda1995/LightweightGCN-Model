# LightweightGCN-Model
A very fast and lightweight model based on Graph Convolutional Network (GCN) for Low Light Image Enhancement (LLIE). It only contains 0.32M parameters; in comparison, if we use a UNet of some depth, then it will take 1.92M parameters. We have not used any pooling layer for down/upsampling our convolution layers, rather, we have used a unique pixelshuffle and inverse pixelshuffle mechanism. Our model is best suited for real-time applications.

To cite: 
```
@article{panda2024integrating,
  title={Integrating Graph Convolution into a Deep Multi-Layer Framework for Low-light Image Enhancement},
  author={Panda, Santosh Kumar and Sa, Pankaj Kumar},
  journal={IEEE Sensors Letters},
  year={2024},
  publisher={IEEE}
}
```


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

**Fig:** Results of our approach, where we compared to the original low light image and our enhanced image.
<hr style="border-top: 3px dotted #998143">



Here I am giving the steps to be followed in Google colab.
1) To run in colab, first of all clone this repo
```
!git clone https://github.com/santoshpanda1995/LightweightGCN-Model.git
```
The requirements.txt file contains all the library files needed for the program to run.

2) Then copy the contents of requirements.txt and just run in a cell in colab.

**Note:** There may be some extra or repeated library files present in the requirements file, usually I maintain a single requirement file so sometimes i copy the existing command which may already be there. It will never effect anything to the program.

3) We have used pixelshuffle and inverse pixelshuffle mechanism for down/upsampling.

4) After that we have to load our trained model, you can directly download the pretrained model which I have provided and put it in the colab to import it.
- [x] LightweightGCN.h5 ( Our pretrained model, trained on our synthetic paired dataset, for 1000 epochs with 32 steps per epoch. Adam Optimizer and L1 loss function has been used.)
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


The above is published in IEEE Sensors Letters, to cite :
```
@article{panda2024integrating,
  title={Integrating Graph Convolution into a Deep Multi-Layer Framework for Low-light Image Enhancement},
  author={Panda, Santosh Kumar and Sa, Pankaj Kumar},
  journal={IEEE Sensors Letters},
  year={2024},
  publisher={IEEE}
}
```
