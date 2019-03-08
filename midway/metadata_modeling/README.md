## preprocessing.py
This .py contains all the helper functions used to load the source data and generate a bag of words feature vector 
for the train and test set saved as as a .npy (numpy pickle). This current script is not executable because the source and output data could not be pushed to GitHub due to the size (> 2 GBs).
## lr_tf.py
This script trains a logistics regression model given a train and test dataset using TensorFlow.  Three folders
should be present in the current working directory: </br>
* data/
* results/
* models/
  
Within the data directory the following files should be present:
</br>
* train_brand_design_mtx_{CLASSES}.npy
* train_one_hot_labels_{CLASSES}.npy
* test_brand_design_mtx_{CLASSES}.npy
* test_one_hot_labels_{CLASSES}.npy
  
Where {CLASSES} is a global definition signifying the number of classes in the target variable (midway report modeled 3 and 58 classes). After running the script, two outputs will be generated: </br>
* /results/lr_brand_by_epoch.npy : numpy pickle of cross entropy and accuracy, by epoch, on the train and test set
* /models/lr_brand_{CLASSES} : saved TensorFlow model
  
If datasets are needed please contact contributor at: angelh@andrew.cmu.edu.


  



