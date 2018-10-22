# EC601-MiniProject2-DeepLearning
This project aims at buliding *training, testing and validation sets* to recgonize between two classes of objects. I implenmented two models the first is to classify cars and trucks and the other is to classify cats and dogs.The tools used are
* Tensorflow: A machine-learning framework to train and validate dataset
* Numpy: A library of array to to load the image data
* Matplotlib: import pyplot to visualize the testing result

## Models
### Cars and Trucks
* The model is demonstrated in carsvstrucks.py. We have two classes of dataset which are cars and trucks. There are 392 images of cars in the folder named *car* and 396 images of trucks in the folder named *truck* 
* I split the dataset using model-selection from sklearn to set training data(80%) and testing data(20%). 
* I use the tf.keras to build CNN models. After 5 epoches of training, the test accuracy is around 74%.(Beacause the dataset is limited, the accuracy is not high enough)Then, we predict the test data to do the validation and visualize the result.

### Cats and Dogs
* The dataset is downloaded from kaggle. It has 25000 images in the *train* folder which have been tagged as cat or dog already and 12500 images in the *test* folder without tagging.
* I use tflearn to build the CNN layers, I build four convolution layers of relu and two fully-connected layers. I set the training rate as 1e-4 and the arruracy of the model is around 80%(The accuracy is higher than the first model). I store the traning and testing data as npy. The model is demonstated in catsvsdogs.py.

## The User Api
* If you want to classify some images, for example 10 images of cats and dogs. Just put them in the front of the *test* folder. In catsvsdogs.py, in the last line, write:*plot_image(2,5)* and you will get the result.
* If you want to classify cars and trucks,put 10 images in either the folder of cars or trucks and they will be automatically generated as test set and you will also get the result by writing *demo(2,5)*
