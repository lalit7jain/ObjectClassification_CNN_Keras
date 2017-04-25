
## Final Exam - Spring 2017

## Contributors
- Lalit Jain https://www.linkedin.com/in/lalit7jain/ 
- Rohit Agarwal https://www.linkedin.com/in/rohitag04/ 
- Shivam Goel https://www.linkedin.com/in/goelshi/
- Hina Gandhi: https://www.linkedin.com/in/hina-gandhi-52834356


>This Assignment consist of Object Classification Using CNN

## Running My Code

- Clone the Repository to System.[Github link](https://github.com/Lalit7Jain/ObjectClassification_CNN_Keras)
- Create Google API key[Google Api Key](https://console.developers.google.com/apis/dashboard?project=final-project-152101)
- Create Gpu Instance Using Amazon Aws Service and Install Django and related libraries to run Python [Link of Document for Reference](https://github.com/Lalit7Jain/ObjectClassification_CNN_Keras/blob/master/Document/Steps%20for%20running%20code.docx)
- SCP the Django files with name mysite  in an AWS GPU Instance under directory '/home/Ubuntu'
> Run the Server using below command
```
python manage.py runserver 0.0.0.0:8000

```

## Data Set 
```
We have dataset of Natural Calamity Namely Hurricane, Volcanic Eruption, Earthquake, Tornado. The dataset of images was collected using Google API. We collected dataset of 400 images equally distributed among 4 classes from google which are jpeg images and relevant to project All images had 3 channels, Red, Green and Blue and were 150x150 pixels large. Figure 1 shows sample data from our collected dataset. There were many grayscale images that would most likely only introduce noise to our model, as such, we had to filter them out.
Additionally, many pictures did not clearly correspond to the class presented. There was lot of corrupted images while downloading from API where were filtered out at later stages before preprocessing starts.
```

- The image data-set is available at [Object Classification Dataset](https://github.com/Lalit7Jain/ObjectClassification_CNN_Keras/tree/master/Django%20code/Django_CNN_Website/mysite/Data)

## Preprocessing

*We preprocessed the images to increase the accuracy of our models. The technique that we used were data augmentation on the dataset to make a new dataset.*

```
A CNN is a neural network that typically contains several types of layers

 convolutional layer
 pooling layer
 activation layers

The final architecture retained can be described as follows:
 • 3 × 3 Conv - ReLU - 2×2 Max-Pool with 32 filters
 • 3 × 3 Conv - ReLU - 2×2 Max-Pool with 32 filters 
 • 3 × 3 Conv - ReLU - 2×2 Max-Pool with 64 filters
 • FC layer to 4 class 


 ```
