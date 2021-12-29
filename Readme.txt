
# Version Required

numpy version : 1.19.5
pandas version : 1.1.5
matplotlib version : 3.2.2
sklearn version : 1.0.1
idx2numpy version : 1.2.3
seaborn version : 0.11.2
torch version : 1.10.0+cu111
torchvision.__version__ : 0.10.1 
gzip
tqdm



# Running the codes


The code will run with the default parameters. Now if you need to change the parameters. Open the .py files.
Go down and change the two main variables 


######### MNIST Classification ###############

In the python files mnist_classification_nondeep.py and mnist_classification_deep.py, in the code just edit and give the paths of the below files.

#train path
train_img_path = 'train-images-idx3-ubyte'
train_label_path = 'train-labels-idx1-ubyte'

# #test path
test_img_path = 't10k-images-idx3-ubyte'
test_label_path = 't10k-labels-idx1-ubyte'

Once the path is edited, the code will run as it is with the default parameters. ### mnist_classification_nondeep.py contains the script for all Non-deep learning methods and results. #### mnist_classification_deep.py contains the CNN Classification for MNIST. Note, you can change other things like seed, no of iterations etc which are well commented. Just need to open the .py file and make the changes.



######### Monkey Classification ###############


In the python files monkey_classification_without_transfer_learning.py and , in the code just edit and give the paths of the below files and set the tr_path and te_path

tr_path = '10-monkey-species/training/training/'
te_path = '10-monkey-species/validation/validation/'


#### monkey_classification_without_transfer_learning.py contains the code for the custom CNN trained on the Monkey species classification dataset. 
#### monky_classification_with_transfer_learning.py contains the code for the custom Resnet-18 finetuned on the Monkey species classification dataset. 


####### Changing the path is important as mentioned ##################
CNN codes will need GPU for fast running.
