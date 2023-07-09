"""
Train a new network on a data set with `train.py`

* Basic usage: `python train.py data_directory`
* Prints out training loss, validation loss, and validation accuracy as the network trains
* Options: 
    * Set directory to save checkpoints: `python train.py data_dir --save_dir save_directory` 
    * Choose architecture: `python train.py data_dir --arch "vgg13"` 
    * Set hyperparameters: `python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20` 
    * Use GPU for training: `python train.py data_dir --gpu`
"""
# imports
import helper
from torch import nn, optim
from tempfile import TemporaryDirectory
from os import path
import torch


def main():
    # define command line argument parsing
    
    # parse command line arguments
    in_args = helper.get_train_input_args()

    # load the data
    dataloader, class_to_idx = helper.load_data(in_args.data_dir)

    # define the device to use for training
    device = helper.get_device(in_args.gpu)
    
    # build pretrained model by calling the build_from_pretrained function
    model = helper.build_from_pretrained(arch= in_args.arch, input_size = in_args.input_size, output_size=in_args.output_size, hidden_units=in_args.hidden_units , drop_p=in_args.drop_p)

    ## Training the network

    # define loss function and optimizer
    criterion = nn.NLLLoss()

    # only train the classifier parameters, feature parameters are frozen
    optimizer = optim.Adam(model.get_submodule(model.extras['classifier_layer']).parameters(), lr=in_args.learning_rate)

    # move model to device
    model.to(device)
    # optimizer.to(device)

    epochs = in_args.epochs
    prev_loss = 1000.
    prev_valid_loss = 1000.
    prev_accuracy = 0.
    best_valid_loss = 1000.


    #parameters for early stopping
    epochs_no_improve = 0
    max_epochs_stop = 5


    arrow_up = '\u2191'
    arrow_down = '\u2193'
    # create a temporary directory to save the best model checkpoints to restore when early stopping
    # Create a temporary directory to save training checkpoints
    with TemporaryDirectory() as tempdir:
        best_weights = path.join(tempdir, 'best_weights.pt')

        #save the current weights as the best weights before training
        torch.save(model.state_dict(), best_weights)

        # train the network
        for epoch in range(epochs):

            # turn on training mode
            model.train()
            running_loss = 0
            steps = 0
            # length of the training dataloader
            train_datalen = len(dataloader['train'])
            # progress_bar = tqdm(total=len(dataloader['train']), leave=False)
            for images, labels in dataloader['train']:
                steps += 1
                # print(images.shape)
                # images = images.view(images.shape[0], 3, -1)
                # move images and labels to device
                images, labels = images.to(device), labels.to(device)
                
                # train
                optimizer.zero_grad()
                logps = model.forward(images)
                loss = criterion(logps, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                # update the progress bar
                # progress_bar.update(1)
                #print progress batch
                print(f'Processing training batch: {steps}/{train_datalen}', end='\r')

                # if steps % 5 == 0:
                #     print(f"Training loss for {steps}/{train_datalen} steps: {running_loss/steps}")
            
            # else:
            # print training loss for each epoch and print up/down if the loss is increasing or decreasing
            print(f"Training loss for epoch {epoch+1}/{epochs}: {running_loss/train_datalen:.5f} {arrow_up if prev_loss < running_loss else arrow_down}")
            prev_loss = running_loss
            # close the progress bar
            # progress_bar.close()

            ##################################
            # validate the network every epoch to check if the model is overfitting
            # stop training if the validation loss does not improve for max_epochs_stop times
            # if epoch % 5 == 0:
            valid_loss = 0
            accuracy = 0
            steps = 0
            valid_datalen = len(dataloader['valid'])
            model.eval()
            with torch.no_grad():
                # progress bar for validation
                # progress_bar = tqdm(total=len(dataloader['valid']), leave=False)

                for images, labels in dataloader['valid']:
                    steps += 1
                    images, labels = images.to(device), labels.to(device)
                    logps = model.forward(images)
                    loss = criterion(logps, labels)
                    valid_loss += loss.item()

                    # calculate accuracy
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    # update the progress bar
                    # progress_bar.update(1)
                    print(f'Processing validation batch: {steps}/{valid_datalen}', end='\r')
            
            # print validation loss and accuracy, also print up/down arrow if the loss/accuracy is increasing or decreasing
            print(f"Validation loss for epoch {epoch+1}/{epochs}: {valid_loss/valid_datalen:.5f} {arrow_up if prev_valid_loss < valid_loss else arrow_down}")
            print(f"Accuracy for epoch {epoch+1}/{epochs}: {100*accuracy/valid_datalen:.2f}% {arrow_up if prev_accuracy < accuracy else arrow_down}")
            print("--------------------------------------------------")
            # update the previous loss and accuracy
            prev_valid_loss = valid_loss
            prev_accuracy = accuracy
            # close the progress bar
            # progress_bar.close()

            # early stopping
            if valid_loss < best_valid_loss:
                epochs_no_improve = 0
                best_valid_loss = valid_loss
                # save the best model weights
                torch.save(model.state_dict(), best_weights)
            else:
                epochs_no_improve += 1

                

            if epochs_no_improve == max_epochs_stop:
                print(f'Early stopping since validation loss did not improve for {max_epochs_stop} epochs.')
                # load the best model weights
                model.load_state_dict(torch.load(best_weights))
                break

            

    #test the network on the test set
    # Do test on the test dataset
    # put model to device mode
    model.to(device)
    model.eval()
    test_loss = 0
    accuracy = 0
    steps = 0
    test_datalen = len(dataloader['test'])

    with torch.no_grad():
        for images, labels in dataloader['test']:
            steps += 1
            
            images, labels = images.to(device), labels.to(device)
            logps = model.forward(images)
            loss = criterion(logps, labels)
            test_loss += loss.item()

            # Calculate accuracy
            ps = torch.exp(logps)
            top_ps, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

            #print progress batch
            print(f'Processing test batch: {steps}/{test_datalen}', end='\r')


        print(f"Test loss: {test_loss/test_datalen}")
        print(f"Accuracy: {accuracy/test_datalen}")
        

    # Save the checkpoint
    model.extras['class_to_idx'] = class_to_idx

    checkpoint = {
        'model_extras': model.extras,
        'state_dict': model.state_dict(),
        'epochs': epoch,
        'optimizer_state_dict': optimizer.state_dict(),
        'optimizer': optimizer,
    }
    torch.save(checkpoint, path.join(in_args.save_dir,'checkpoint.pth'))








# if the program is run from the command line
if __name__ == "__main__":
    main()