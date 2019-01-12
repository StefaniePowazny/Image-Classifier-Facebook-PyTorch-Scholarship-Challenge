# Imports here
import numpy as np
import torch
from torchvision import datasets
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, models
import os
import json
from collections import OrderedDict
from torch.autograd import Variable
from PIL import Image
import matplotlib.pyplot as plt

# check if CUDA is available
train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('CUDA is not available. Training on CPU >.<')
else:
    print('CUDA is available! Training on GPU ^_^')

# load the data
data_dir = os.path.join(".", "flower_data")
train_dir = os.path.join(data_dir, "train")
valid_dir = os.path.join(data_dir, "valid")

# TODO: Define your transforms for the training and validation sets
size = 224

train_transforms = transforms.Compose([
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(45),
    transforms.RandomResizedCrop(size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

validation_transforms = transforms.Compose([transforms.Resize(256),
                                            transforms.CenterCrop(size),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

# TODO: Load the datasets with ImageFolder
train_data = datasets.ImageFolder(train_dir, transform=train_transforms)

validation_data = datasets.ImageFolder(valid_dir, transform=validation_transforms)

# TODO: Using the image datasets and the trainforms, define the dataloaders
batch_size = 128
num_workers = 0

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                           num_workers=num_workers, shuffle=True)
validation_loader = torch.utils.data.DataLoader(validation_data, batch_size=batch_size,
                                                num_workers=num_workers)

class_to_idx = train_data.class_to_idx

# label mapping
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

print(F"There are {len(cat_to_name)} categories.")

# look at an image
images, labels = next(iter(train_loader))
print(len(images[0, 2]))
plt.imshow(images[0, 0])

# load a pre-trained model
model = models.densenet121(pretrained=True)
print(model)

# freeze
for param in model.parameters():
    param.requires_grad_(False)

# build the classifier
classifier_input_size = model.classifier.in_features
n_hidden_layer = (classifier_input_size // 2)
n_classes = 102

# TODO: zweites dropout weg, vielleicht ein layer weniger
classifier = nn.Sequential(OrderedDict([
    ('dropout', nn.Dropout(p=0.20)),
    ('fc1', nn.Linear(classifier_input_size, n_hidden_layer)),
    ('relu1', nn.ReLU()),
    ('fc2', nn.Linear(n_hidden_layer, n_classes)),
    ('output', nn.LogSoftmax(dim=1))]))

model.classifier = classifier
model.class_to_idx = class_to_idx

print(model)

# move the model to GPU, if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model.to(device)

# https://discuss.pytorch.org/t/runtimeerror-expected-object-of-type-torch-floattensor-but-found-type-torch-cuda-floattensor-for-argument-2-weight/27483
# https://discuss.pytorch.org/t/how-to-use-dataparallel-in-backward/3556
# https://github.com/sedemmler/AIPND_image_classification/blob/master/Image%20Classifier%20Project.ipynb
# https://github.com/marvis/pytorch-yolo2/issues/89

# specify loss function (categorical cross-entropy)
criterion = nn.NLLLoss()

# specify optimizer
optimizer = torch.optim.Adadelta(model.classifier.parameters())

# number of epochs to train the model
n_epochs = 15

valid_loss_min = np.Inf  # track change in validation loss

for epoch in range(n_epochs):

    # keep track of training and validation loss
    train_loss = 0
    valid_loss = 0
    accuracy = 0

    ###################
    # train the model #
    ###################
    model.train()
    for ii, (inputs, labels) in enumerate(train_loader):
        # use gpu
        inputs, labels = inputs.to(device), labels.to(device)
        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model.forward(inputs)
        # calculate the batch loss
        loss = criterion(output, labels)
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()
        # update training loss
        # train_loss += loss.item()*inputs.size(0)
        train_loss += loss.item()

    ######################
    # validate the model #
    ######################
    model.eval()
    for ii, (inputs, labels) in enumerate(validation_loader):
        # use gpu
        inputs, labels = inputs.to(device), labels.to(device)
        # if torch.cuda.is_available():
        #    inputs = Variable(inputs.float().cuda())
        #   labels = Variable(labels.long().cuda())
        # else:
        #   inputs = Variable(inputs)
        #  labels = Variable(labels)
        with torch.no_grad():
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model.forward(inputs)
            # calculate the batch loss
            loss = criterion(output, labels)
            # update average validation loss
            # valid_loss += loss.item()*inputs.size(0)
            valid_loss += loss.item()
            # accuracy
            # ps = torch.exp(output).data
            # equality = (labels.data == ps.max(1)[1])
            # accuracy += equality.type_as(torch.FloatTensor()).mean()
            ps = torch.exp(output)
            equality = (labels.data == ps.max(dim=1)[1])
            accuracy += equality.type(torch.FloatTensor).mean()

    # calculate average losses
    train_loss = train_loss / len(train_loader)
    valid_loss = valid_loss / len(validation_loader)
    accuracy = accuracy / len(validation_loader)

    # print training/validation statistics
    print('Epoch: {}/{} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f} \tValidation Accuracy: {:.6f}'.format(
        epoch + 1, n_epochs, train_loss, valid_loss, accuracy))

    # state if validation loss has decreased
    if valid_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}). Wuhuuu \(^o^)/ '.format(
            valid_loss_min,
            valid_loss))
        valid_loss_min = valid_loss

# TODO: Save the checkpoint
# https://github.com/fotisk07/Image-Classifier/blob/master/Image%20Classifier%20Project.ipynb
torch.save({
    'architecture': 'densenet121',
    'hidden_layers': n_hidden_layer,
    'classifier': classifier,
    'epochs': n_epochs,
    'state_dict': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'class_to_idx': model.class_to_idx},
    'checkpoint.pth')

# load the checkpoint
checkpoint = torch.load('checkpoint.pth')
print(checkpoint['architecture'])


# https://stackoverflow.com/questions/3061/calling-a-function-of-a-module-by-using-its-name-a-string
def load_model(path):
    checkpoint = torch.load(path)
    model_name = checkpoint['architecture']
    model = getattr(models, model_name)(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    classifier = checkpoint['classifier']
    model.classifier = classifier
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    # move the model to GPU, if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    return model


model = load_model('checkpoint.pth')
print(model)


def process_image(image):
    """ Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    """
    pil_image = Image.open(image)

    pil_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(45),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    pil_image = pil_transforms(pil_image)

    return pil_image

# TODO: Process a PIL image for use in a PyTorch model
image = (train_dir + '/1/' + 'image_06750.jpg')
image = process_image(image)
print(image.shape)

def imshow(image, ax=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()

    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))

    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)

    ax.imshow(image)

    return ax


imshow(image)

# class prediction

# https://github.com/bryanfree66/AIPND_image_classification/blob/master/Image%20Classifier%20Project.ipynb
def predict(image_path, model, topk=5):
    """ Predict the class (or classes) of an image using a trained deep learning model.
    """
    model.eval()
    # move the model to GPU, if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    # process images, unsqueeze returns a new tensor with a dimension of size one
    pil_image = process_image(image_path)
    tensor = pil_image.unsqueeze_(0)
    # tensor = tensor.float()

    with torch.no_grad():
        output = model.forward(tensor.cuda())
    ps = torch.exp(output).data.topk(topk)
    probs = ps[0].cpu()
    classes = ps[1].cpu()
    class_to_idx_inverted = {model.class_to_idx[k]: k for k in model.class_to_idx}
    mapped_classes = list()
    for i in classes.numpy()[0]:
        mapped_classes.append(class_to_idx_inverted[i])

    return probs.numpy()[0], mapped_classes

    # TODO: Implement the code to predict the class from an image file
image_path = (train_dir + '/1/' + 'image_06750.jpg')

probs, classes = predict(image_path, model)
print(probs)
print(classes)

# sanity checking

correct_label = str(image_path)
print(correct_label.split('/')[1])

# TODO: Display an image along with the top 5 classes
image_path = (train_dir + '/1/' + 'image_06750.jpg')


def sanity_check(image_path, model):
    probs, classes = predict(image_path, model)
    ax1 = plt.subplot2grid((15, 9), (0, 0), colspan=9, rowspan=9)
    ax2 = plt.subplot2grid((15, 9), (9, 2), colspan=5, rowspan=5)

    # get the correct label
    correct_label = str(image_path)
    correct_label = correct_label.split('/')[1]
    image = Image.open(image_path)
    ax1.axis('off')
    ax1.set_title(cat_to_name[correct_label])
    ax1.imshow(image)
    labels = []
    for i in classes:
        labels.append(cat_to_name[i])
    y_pos = np.arange(5)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(labels)
    ax2.invert_yaxis()  # probabilities read top-to-bottom
    ax2.set_xlabel('Probability')
    ax2.barh(y_pos, probs, xerr=0, align='center')

    plt.show()


sanity_check(image_path, model)