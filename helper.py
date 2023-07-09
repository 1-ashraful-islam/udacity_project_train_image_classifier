import argparse
import torch
from torchvision import datasets, transforms, models
from torch import nn, optim
from collections import OrderedDict
from PIL import Image
import numpy as np


# create a argument parser for train.py
def get_train_input_args():
    # Specifications:
    # Set directory to save checkpoints: python train.py data_dir --save_dir save_directory
    # Choose architecture: python train.py data_dir --arch "vgg13"
    # Set hyperparameters: python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
    # Use GPU for training: python train.py data_dir --gpu
    
    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser()
    # Create command line arguments as mentioned above using add_argument() from ArguementParser method
    parser.add_argument('data_dir', type=str, default='flower_data', help='path to folder of images')
    parser.add_argument('--save_dir', type=str, default='', help='directory to save checkpoints')
    parser.add_argument('--arch', type=str, default='vgg19', help='pretrained model architecture to use for network',
        choices=['vgg19', 'resnet50', 'inception_v3','resnext101_32x8d'])
    parser.add_argument('--learning_rate', type=float, default=0.001, help='hyperparameter: learning rate')
    parser.add_argument('--hidden_units', type=list, default=[2048, 1024, 512], help='hyperparameter: number of hidden units')
    parser.add_argument('--epochs', type=int, default=20, help='hyperparameter: number of epochs')
    parser.add_argument('--drop_p', type=float, default=0.5, help='hyperparameter: dropout probability')
    parser.add_argument('--input_size', type=int, default=4096, help='hyperparameter: input size')
    parser.add_argument('--output_size', type=int, default=102, help='hyperparameter: output size')
    parser.add_argument('--gpu', action='store_true', help='use GPU for training')

    return parser.parse_args()

# create a argument parser for predict.py
def get_predict_input_args():
    # Specifications:
    # Basic usage: python predict.py /path/to/image checkpoint
    # Options:
    # Return top K most likely classes: python predict.py input checkpoint --top_k 3
    # Use a mapping of categories to real names: python predict.py input checkpoint --category_names cat_to_name.json
    # Use GPU for inference: python predict.py input checkpoint --gpu
    
    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser()
    # Create command line arguments as mentioned above using add_argument() from ArguementParser method
    parser.add_argument('image_path', type=str, default='assets/prink_primrose_inference_test.jpg', help='path to image file')
    parser.add_argument('checkpoint', type=str, default='checkpoint.pth', help='path to checkpoint')
    parser.add_argument('--top_k', type=int, default=1, help='return top K most likely classes')
    parser.add_argument('--category_names', type=str, default='', help='mapping of categories to real names')
    parser.add_argument('--gpu', action='store_true', help='use GPU for inference')
    
    return parser.parse_args()


# load a checkpoint and rebuild the model
def load_checkpoint(filepath):
    """
    Loads a checkpoint and rebuilds the model
    Input: filepath to the checkpoint
    Output: model, epoch, optimizer
    """

    checkpoint = torch.load(filepath)
    
    # load the checkpoint extras
    model_extras = checkpoint.get('model_extras')
    if model_extras is None:
        raise Exception("The checkpoint file does not contain the information needed to rebuild the model")
    
    # print(model_extras)
    
    # if it contains the information needed to rebuild the model, then rebuild the model
    # since we are loading the model from a checkpoint if any of the model_extras are not provided
    # then the rebuild will fail
    arch = model_extras.get('arch')
    input_size = model_extras.get('input_size')
    output_size = model_extras.get('output_size')
    hidden_units = model_extras.get('hidden_units')
    drop_p = model_extras.get('drop_p')

    # check if any of values are None
    if None in [arch, input_size, output_size, hidden_units, drop_p]:
        # print which values are None
        # print(f"arch: {arch}, input_size: {input_size}, output_size: {output_size}, hidden_units: {hidden_units}, drop_p: {drop_p}")
        raise Exception("The checkpoint file does not contain the information needed to rebuild the model")

    # rebuild the model
    model = build_from_pretrained(
        arch=arch,
        input_size=input_size,
        output_size=output_size,
        hidden_units=hidden_units,
        drop_p= drop_p
        )
    
    # load the class to idx mapping
    model.extras['class_to_idx'] = model_extras.get('class_to_idx')
    # load the model state dict (weights and biases) and the optimizer state dict
    model.load_state_dict(checkpoint.get('state_dict'))
    epoch = checkpoint.get('epochs')

    optimizer = checkpoint.get('optimizer')
    optimizer.load_state_dict(checkpoint.get('optimizer_state_dict'))

    return model, optimizer, epoch


# process a PIL image for use in a PyTorch model
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    with Image.open(image) as im:
        # resize the image keeping the shorter side 256 pixels
        width, height = im.size
        if width > height:
            im.thumbnail((10000, 256))
        else:
            im.thumbnail((256, 10000))
        
        # center crop the image to 224x224 pixels
        width, height = im.size
        left = (width - 224)/2
        top = (height - 224)/2
        right = (width + 224)/2
        bottom = (height + 224)/2
        im = im.crop((left, top, right, bottom))

        # convert image to numpy array
        np_image = np.array(im)
        # convert to 0-1 scale from 0-255
        np_image = np_image/255
        # normalize the image with means [0.485, 0.456, 0.406] and standard deviation [0.229, 0.224, 0.225]
        np_image = (np_image - [0.485, 0.456, 0.406])/[0.229, 0.224, 0.225]
        # transpose the image so the color channel is the first dimension
        np_image = np_image.transpose((2, 0, 1))
        return torch.from_numpy(np_image).type(torch.FloatTensor)

#get device function when using gpu or cpu
# this function lets you get gpu in mac without specifying cuda as the device
def get_device(gpu):
    # select GPU if available else CPU
    if gpu:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            print("MPS device found. Switching to MPS.")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
            print("CUDA device found. Switching to GPU.")
    else:
        device = torch.device("cpu")
    return device


# build from pre-trained
def build_from_pretrained(arch = 'vgg19', input_size = 4096, output_size = 102, hidden_units = [4096, 500], drop_p = 0.5):
    # dict of models that can be used in this project
    model_dict = {
        'vgg19': 'classifier',
        'resnet50': 'fc',
        'inception_v3': 'fc',
        'resnext101_32x8d': 'fc',
    }


    # Load a pre-trained network
    # arch = 'vgg19'
    if model_dict.get(arch) is None:
        print(f'Error: {arch} is not a supported model architecture')
        # print the valid model names
        print('Currently supported model architectures are:')
        for key in model_dict.keys():
            print(key)
        # select a default valid model name
        arch = list(model_dict.keys())[0]
        print(f'Using {arch} as the default model architecture')


    model = models.get_model(arch, pretrained=True)

    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    # pass the classifer layer and other input arguments as attribute to the model to use it later
    # save the input params 
    extras = {
        'arch': arch,
        'input_size': input_size,
        'output_size': output_size,
        'hidden_units': hidden_units,
        'drop_p': drop_p,
        'classifier_layer': model_dict[arch],
    }
    # add the extras as attribute to the model to use it later in checkpoints
    model.extras = extras
    

    # get the input size of the classifier from the pretrained model
    # print(f'Input size of the classifier is {model_dict[arch]} {') 
    # get classifier layer
    classifier_layer = getattr(model, model_dict[arch])
    # check if the classifier layer is a sequential layer or Linear layer
    # if it is a sequential layer then get the input size of the first layer
    # if it is a Linear layer then get the input size of the layer
    if isinstance(classifier_layer, nn.Sequential):
        model_classifier_input_size = classifier_layer[0].in_features
    elif isinstance(classifier_layer, nn.Linear): 
        model_classifier_input_size = classifier_layer.in_features
    # check if the input size is the same as the input size of the classifier otherwise print a error
    # and select the model input size as the input size of the classifier
    # if model_classifier_input_size != input_size:
    #     print(f'Error: The input size of the classifier is {model_classifier_input_size} but the input size you specified is {input_size}')
    #     print(f'Using {model_classifier_input_size} as the input size of the classifier to avoid errors')
    #     input_size = model_classifier_input_size

    # Build a feed-forward classifier network for arbitrary number of hidden layers
    # add interface layer to avoid  errors when input size is not the same as the input size of the original model classifier
    classifier = nn.Sequential(OrderedDict([
        ('interface', nn.Linear( model_classifier_input_size, input_size)),
        ('relu_interface', nn.ReLU()),
        ('dropout_interface', nn.Dropout(p=drop_p)),
        ('input',nn.Linear(input_size, hidden_units[0])),
        ('relu0',nn.ReLU()),
        ('dropout0',nn.Dropout(p=drop_p)),
    ])
    )
    # add hidden layers
    for i in range(len(hidden_units)-1):
        classifier.add_module(f'fc{i+1}', nn.Linear(hidden_units[i], hidden_units[i+1]))
        classifier.add_module(f'relu{i+1}', nn.ReLU())
        classifier.add_module(f'dropout{i+1}', nn.Dropout(p=0.2))


    # add output layer
    classifier.add_module('output',nn.Linear(hidden_units[-1], output_size))
    classifier.add_module('logps',nn.LogSoftmax(dim=1))
    
    # print(model)
    # replace the classifier in the model
    setattr(model, model_dict[arch], classifier)
    # # remove the classifier or fc layer in the model
    # delattr(model, model_dict[arch])

    return model

# load data
def load_data(data_dir = 'flower_data', batch_size = 64):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # Define your transforms for the training, validation, and testing sets

    train_transforms = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]),
        ])
    valid_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]),
    ])

    test_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]),
    ])

                                                                

    #Load the datasets with ImageFolder
    train_datasets = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_datasets = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    test_datasets = datasets.ImageFolder(test_dir, transform=test_transforms)

    # Using the image datasets and the trainforms, define the dataloaders
    train_dataloaders = torch.utils.data.DataLoader(train_datasets, batch_size=batch_size, shuffle=True)
    valid_dataloaders = torch.utils.data.DataLoader(valid_datasets, batch_size=batch_size, shuffle=True)
    test_dataloaders = torch.utils.data.DataLoader(test_datasets, batch_size=batch_size, shuffle=True)

    dataloaders = {
        'train': train_dataloaders,
        'valid': valid_dataloaders,
        'test': test_dataloaders,
    }
    return dataloaders, train_datasets.class_to_idx