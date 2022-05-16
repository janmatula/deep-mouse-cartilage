#author Jan Matula
#created on 30/03/22
import os
import random
import numpy as np
from PIL import Image
import tensorflow as tf
import time
from model_residual_selu_deeplysup import residual_selu_deeplysup
from IPython.display import clear_output
import pickle


#path to data and hyperparameters
path_to_data = "../data/Sample_slices"
name = "example_trained_model"
batch_size = 4 #batch size
epochs = 10 #number of training epochs

#load the model, this can either be a randomly initialized model or already trained model for transfer learning
model = residual_selu_deeplysup((1280, 1792, 1), n_first_layer = 4)

#load paths to images and masks
paths_to_images = sorted([os.path.join(os.path.join(path_to_data, "Images").replace("\\","/"), f).replace("\\","/") for f in os.listdir(os.path.join(path_to_data, "Images").replace("\\","/")) if f.endswith(".tif")])
paths_to_masks = sorted([os.path.join(os.path.join(path_to_data, "Ground_truth").replace("\\","/").replace("\\","/"), f) for f in os.listdir(os.path.join(path_to_data, "Ground_truth").replace("\\","/")) if f.endswith(".tif")])

#create experiment name and create folder
experiment_name = name

try:
    os.mkdir("../results/Example_trained_model/")
except:
    print("Folder already exists.")

try:
    os.mkdir("../results/Example_trained_model/" + experiment_name)
except:
    print("Folder already exists.")

#divide images into train and validation sets
indices = list(range(0, len(paths_to_images)))
random.shuffle(indices)
paths_to_images_train = [paths_to_images[index] for index in indices[:40]]
paths_to_masks_train = [paths_to_masks[index] for index in indices[:40]]
paths_to_images_val = [paths_to_images[index] for index in indices[40:]]
paths_to_masks_val = [paths_to_masks[index] for index in indices[40:]]

#load validation images
x_val = np.expand_dims(np.stack([np.array(Image.open(paths_to_images_val[k])) for k in list(range(0, len(paths_to_images_val)))], axis = 0), axis =-1)
y_val = np.expand_dims(np.stack([np.array(Image.open(paths_to_masks_val[k])) for k in list(range(0, len(paths_to_masks_val)))], axis = 0), axis = -1)

training_loss = [0]
validation_loss = [0]

#training loop
indices = list(range(0, len(paths_to_images_train)))
start = time.time()
for i in range(epochs):
    #random shuffle
    random.shuffle(indices)
    epoch_loss = []
    for j in range(0,  len(paths_to_images_train)-batch_size, batch_size):
        #load batch
        x = np.expand_dims(np.stack([np.array(Image.open(paths_to_images_train[k])) for k in indices[j:(j+batch_size)]], axis = 0), axis =-1)
        y = np.expand_dims(np.stack([np.array(Image.open(paths_to_masks_train[k])) for k in indices[j:(j+batch_size)]], axis = 0), axis = -1)
        #train on batch
        epoch_loss.append(model.train_on_batch(x, [y for n in range(0,7)])[0])
        clear_output(wait=True)
        print("Epoch " + str(i+1) + ", Dice: " + str(-epoch_loss[-1]*100) + " %", flush = True)

    training_loss.append(np.mean(epoch_loss))
    #if training loss improves, save checkpoint
    if training_loss[-1]<=np.min(training_loss):
        model.save("../results/Example_trained_model/"+ experiment_name + "/best_training_model.h5")
        best_training_epoch_number = i+1

    #validate on validation set
    validation_loss.append(model.evaluate(x_val, [y_val for i in range(0, 7)], batch_size=4)[0])

    #if validation loss improves, save validation checkpoint
    if validation_loss[-1]<=np.min(validation_loss):
        model.save("../results/Example_trained_model/" + experiment_name + "/best_validation_model.h5")
        best_validation_epoch_number = i+1
    #save the final mode
    model.save("../results/Example_trained_model/" + experiment_name + "/latest_model.h5")

end = time.time()
elapsed_time = end-start
print("The training finished in " + '{0:.2f}'.format(elapsed_time) + " seconds!")

#save the information about the experiment
file = open("../results/Example_trained_model/" + experiment_name + "/" + experiment_name + ".txt", "w") 
file.write("Batch size: " + str(batch_size) + "\n")
file.write("Epochs: " + str(epochs) + "\n")
file.write("Best training epoch: " + str(best_training_epoch_number) + "\n")
file.write("Best validation epoch: " + str(best_validation_epoch_number) + "\n")
file.write("Trained on:\n")
file.write('\n'.join([os.path.basename(os.path.normpath(p)) for p in paths_to_images_train]) + '\n')
file.write("Validated on on:\n")
file.write('\n'.join([os.path.basename(os.path.normpath(p)) for p in paths_to_images_val]) + '\n')
file.write('\nElapsed time [s] '+  '{0:.2f}'.format(elapsed_time) + '\n')
file.close() 

#save training and validation losses
with open("../results/Example_trained_model/" + experiment_name + "/training_loss.txt", "wb") as fp:
    pickle.dump(training_loss, fp)
with open("../results/Example_trained_model/" + experiment_name + "/validation_loss.txt", "wb") as fp:
    pickle.dump(validation_loss, fp)