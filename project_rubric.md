# Udacity Project : Create Your Own Image Classifier
---------------------

## Project Overview


Files Submitted:

For a successful project submission, you'll need to include these files in a ZIP archive (Not sure if we need to submit zip archive if submitted using GitHub):

- <input type="checkbox"> The completed Jupyter Notebook from Part 1 as an HTML file and any extra files you created that are necessary to run the code in the notebook

- <input type="checkbox"> The train.py and predict.py files from Part 2, as well as any other files necessary to run those scripts


====================================

# Project Specifications/Rubric
---------------------

|  | Criteria | Meets Specifications |
| ---| --- | --- |
| <input type="checkbox"> | Submission Files | The submission includes all required files. (Model checkpoints not required.) |

## Part 1 - Development Notebook

|  | Criteria | Meets Specifications |
| --- | --- | --- |
| <input type="checkbox" checked> | Package Imports | All the necessary packages and modules are imported in the first cell of the notebook |
| <input type="checkbox" checked> | Training data augmentation | torchvision transforms are used to augment the training data with random scaling, rotations, mirroring, and/or cropping |
| <input type="checkbox" checked> | Data normalization | The training, validation, and testing data is appropriately cropped and normalized |
| <input type="checkbox" checked> | Data loading | The data for each set (train, validation, test) is loaded with torchvision's ImageFolder |
| <input type="checkbox" checked> | Data batching | The data for each set is loaded with torchvision's DataLoader |
| <input type="checkbox" checked> | Pretrained Network | A pretrained network such as VGG16 is loaded from torchvision.models and the parameters are frozen |
| <input type="checkbox" checked> | Feedforward Classifier | A new feedforward network is defined for use as a classifier using the features as input |
| <input type="checkbox" checked> | Training the network | The parameters of the feedforward classifier are appropriately trained, while the parameters of the feature network are left static |
| <input type="checkbox" checked> | Validation Loss and Accuracy | During training, the validation loss and accuracy are displayed |
| <input type="checkbox" checked> | Testing Accuracy | The network's accuracy is measured on the test data |
| <input type="checkbox" checked> | Saving the model | The trained model is saved as a checkpoint along with associated hyperparameters and the class\_to\_idx dictionary |
| <input type="checkbox" checked> | Loading checkpoints | There is a function that successfully loads a checkpoint and rebuilds the model |
| <input type="checkbox" checked> | Image Processing | The process_image function successfully converts a PIL image into an object that can be used as input to a trained model |
| <input type="checkbox" checked> | Class Prediction | The predict function successfully takes the path to an image and a checkpoint, then returns the top K most probably classes for that image |
| <input type="checkbox" checked> | Sanity Checking with matplotlib | A matplotlib figure is created displaying an image and its associated top 5 most probable classes with actual flower names |

## Part 2 - Command Line Application

|  | Criteria | Meets Specifications |
| --- | --- | --- |
| <input type="checkbox"> | Training a network | train.py successfully trains a new network on a dataset of images and saves the model to a checkpoint |
| <input type="checkbox"> | Training validation log | The training loss, validation loss, and validation accuracy are printed out as a network trains |
| <input type="checkbox"> | Model architecture | The training script allows users to choose from at least two different architectures available from torchvision.models |
| <input type="checkbox"> | Model hyperparameters | The training script allows users to set hyperparameters for learning rate, number of hidden units, and training epochs |
| <input type="checkbox"> | Training with GPU | The training script allows users to choose training the model on a GPU |
| <input type="checkbox"> | Predicting classes | The predict.py script successfully reads in an image and a checkpoint then prints the most likely image class and it's associated probability |
| <input type="checkbox"> | Top K classes | The predict.py script allows users to print out the top K classes along with associated probabilities |
| <input type="checkbox"> | Displaying class names | The predict.py script allows users to load a JSON file that maps the class values to other category names |
| <input type="checkbox"> | Predicting with GPU | The predict.py script allows users to use the GPU to calculate the predictions |\
