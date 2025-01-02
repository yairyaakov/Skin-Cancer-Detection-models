# Skin Cancer Detection models 🩺 | CNN, NN and Logistic Regression

#### Prologue
This notebook explores melanoma classification using machine learning.
The objective is **construct** and **analyze** three models: a neural network, logistic regression, and a CNN.

#### Dataset Overview
Comprising 13,900 uniformly-sized images at 224 x 224 pixels, which provides a comprehensive portrayal of diverse manifestations of melanoma. Each image is meticulously labeled as either `benign` or `malignant`.


```python
import os
import random
#----------
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
#----------
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report, mean_squared_error, accuracy_score, precision_score, recall_score
#----------
import matplotlib.pyplot as plt
#----------
import numpy as np
#----------
import warnings
warnings.filterwarnings('ignore')
```

## Load and preprocess the dataset


```python
# Path to the dataset archive
archive_path = r'parth\to\archive'
```


```python
# Define the main folder path after extraction
main_folder_path = os.path.splitext(archive_path)[0]  # Remove the extension
```

The main folder contains 2 folders - train and test - and each of them conatins 2 folder - Benign and Malignant


```python
# Define subfolders
data_folders = ["train", "test"]
class_folders = ["Benign", "Malignant"]
```


```python
# Paths for train and test data
train_data_path = os.path.join(main_folder_path, data_folders[0])
test_data_path = os.path.join(main_folder_path, data_folders[1])
```

Set those hyperparamaters as you wish


```python
img_width, img_height = 112, 112 
batch_size = 128
epochs = 15
```


```python
# Data generators
train_datagen = ImageDataGenerator(rescale=1.0 / 255)
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_path,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary'
)

test_generator = test_datagen.flow_from_directory(
    test_data_path,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False
)
```

    Found 11879 images belonging to 2 classes.
    Found 2000 images belonging to 2 classes.
    

### Samples Examples


```python
# Get the first batch from the training generator
x_batch, y_batch = next(train_generator)

# Extract features and label from the first element in the batch
first_features = x_batch[0]
first_label = y_batch[0]

# Get the mapping of class indices to class names
class_indices = train_generator.class_indices

# Reverse the mapping to get class names to class indices
class_names = {v: k for k, v in class_indices.items()}

# Extract the numerical label of the first element in the batch
numerical_label = int(first_label)

# Get the corresponding class name
label_name = class_names[numerical_label]
print(int(first_label), "stands for", label_name)
```

    1 stands for Malignant
    


```python
# Display information about the dataset
shapes = np.shape(train_generator[0][0])
print("A batch contains", shapes[0], "samples of", shapes[1], "x", shapes[2], "x", shapes[3])
```

    A batch contains 128 samples of 112 x 112 x 3
    


```python
# Select 3 random indices from the list
random_indices = random.sample(range(len(train_generator)), 3)

# Display the selected images in a 3x1 grid
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

for i, index in enumerate(random_indices):
    # Show each image
    image = train_generator[index][0][0]
    axes[i].imshow(image)
    axes[i].axis('off')

plt.tight_layout()
plt.show()
```


    
![png](pngs/output_15_0.png)
    


# Models

This section involves constructing three models: a Neural Network (NN), Logistic Regression, and Convolutional Neural Network (CNN). Each model is analyzed individually, followed by a comparative evaluation to discern their respective performance characteristics.

## Neural Network 

### Creation


```python
def create_nn(num_hidden_layers, hidden_layer_sizes, learning_rate=0.0001):
    """
    Create a neural network with dynamic hidden layers and a specified learning rate.

    Parameters:
    - num_hidden_layers: Integer specifying the number of hidden layers for each set of sizes.
    - hidden_layer_size: List of integers specifying the size of each hidden layer.
    - learning_rate: Float specifying the learning rate.
    """
    model = Sequential()
    
    # Flatten the input data
    model.add(Flatten(input_shape=(img_width, img_height, 3)))

    # Add hidden layers with dropout
    for i in range(num_hidden_layers):
        model.add(tf.keras.layers.Dense(hidden_layer_sizes[i], activation='relu'))
            
    # Output layer with binary classification
    model.add(Dense(1, activation='sigmoid')) 

    # Compile the model with specified learning rate
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy'])

    return model
```


```python
# Define NN sizes
hidden_layer_sizes = [128, 64, 64]
num_hidden_layers = len(hidden_layer_sizes)

# Get the NN model
nn_model = create_nn(num_hidden_layers, hidden_layer_sizes)

# Display the model architecture
nn_model.summary()
```

    Model: "sequential_7"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     flatten_6 (Flatten)         (None, 37632)             0         
                                                                     
     dense_12 (Dense)            (None, 128)               4817024   
                                                                     
     dense_13 (Dense)            (None, 64)                8256      
                                                                     
     dense_14 (Dense)            (None, 64)                4160      
                                                                     
     dense_15 (Dense)            (None, 1)                 65        
                                                                     
    =================================================================
    Total params: 4829505 (18.42 MB)
    Trainable params: 4829505 (18.42 MB)
    Non-trainable params: 0 (0.00 Byte)
    _________________________________________________________________
    

### Training and evaluating


```python
history = nn_model.fit(train_generator, epochs=epochs, validation_data=test_generator)

# Evaluate the model
test_loss, test_acc = nn_model.evaluate(test_generator)
print(f'Test Accuracy: {test_acc}')
```

    Epoch 1/15
    93/93 [==============================] - 44s 474ms/step - loss: 0.4202 - accuracy: 0.8024 - val_loss: 0.5810 - val_accuracy: 0.6845
    Epoch 2/15
    93/93 [==============================] - 46s 490ms/step - loss: 0.4060 - accuracy: 0.8106 - val_loss: 0.4253 - val_accuracy: 0.8025
    Epoch 3/15
    93/93 [==============================] - 43s 467ms/step - loss: 0.3870 - accuracy: 0.8205 - val_loss: 0.4619 - val_accuracy: 0.7535
    Epoch 4/15
    93/93 [==============================] - 44s 477ms/step - loss: 0.3760 - accuracy: 0.8286 - val_loss: 0.4476 - val_accuracy: 0.7645
    Epoch 5/15
    93/93 [==============================] - 49s 524ms/step - loss: 0.3739 - accuracy: 0.8289 - val_loss: 0.3937 - val_accuracy: 0.8340
    Epoch 6/15
    93/93 [==============================] - 49s 527ms/step - loss: 0.3734 - accuracy: 0.8301 - val_loss: 0.4006 - val_accuracy: 0.8205
    Epoch 7/15
    93/93 [==============================] - 43s 466ms/step - loss: 0.3619 - accuracy: 0.8383 - val_loss: 0.3654 - val_accuracy: 0.8510
    Epoch 8/15
    93/93 [==============================] - 43s 466ms/step - loss: 0.3644 - accuracy: 0.8412 - val_loss: 0.3984 - val_accuracy: 0.8210
    Epoch 9/15
    93/93 [==============================] - 44s 479ms/step - loss: 0.3539 - accuracy: 0.8427 - val_loss: 0.4664 - val_accuracy: 0.7510
    Epoch 10/15
    93/93 [==============================] - 47s 502ms/step - loss: 0.3489 - accuracy: 0.8440 - val_loss: 0.3685 - val_accuracy: 0.8345
    Epoch 11/15
    93/93 [==============================] - 55s 594ms/step - loss: 0.3496 - accuracy: 0.8426 - val_loss: 0.4957 - val_accuracy: 0.7305
    Epoch 12/15
    93/93 [==============================] - 45s 482ms/step - loss: 0.3409 - accuracy: 0.8452 - val_loss: 0.3477 - val_accuracy: 0.8405
    Epoch 13/15
    93/93 [==============================] - 45s 483ms/step - loss: 0.3416 - accuracy: 0.8478 - val_loss: 0.3913 - val_accuracy: 0.8465
    Epoch 14/15
    93/93 [==============================] - 45s 483ms/step - loss: 0.3424 - accuracy: 0.8497 - val_loss: 0.3544 - val_accuracy: 0.8670
    Epoch 15/15
    93/93 [==============================] - 46s 490ms/step - loss: 0.3350 - accuracy: 0.8486 - val_loss: 0.4043 - val_accuracy: 0.8020
    16/16 [==============================] - 6s 368ms/step - loss: 0.4043 - accuracy: 0.8020
    Test Accuracy: 0.8019999861717224
    

### Training Results


```python
# Plot training and test accuracy values
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(np.arange(1, len(history.history['val_accuracy']) + 1), history.history['val_accuracy'], label='Test Accuracy', color="steelblue")
plt.plot(np.arange(1, len(history.history['accuracy']) + 1), history.history['accuracy'], label='Training Accuracy', color="skyblue")
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Test Accuracy')
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(range(1, len(history.history['accuracy']) + 1))
plt.legend()

# Plot training and test loss values
plt.subplot(1, 2, 2)
plt.plot(np.arange(1, len(history.history['val_loss']) + 1), history.history['val_loss'], label='Test Loss', color="steelblue")
plt.plot(np.arange(1, len(history.history['loss']) + 1), history.history['loss'], label='Training Loss', color="skyblue")
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Test Loss')
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(range(1, len(history.history['loss']) + 1))
plt.legend()

plt.tight_layout()
plt.show()
```


    
![png](pngs/output_25_0.png)
    


### Model Evaluation Metrics


```python
# Evaluate the model on the test data and get predictions
predicted_probabilities = nn_model.predict(test_generator)

# Convert probabilities to binary predictions (0 or 1)
predicted_labels = np.round(predicted_probabilities).astype(np.int32)

# Get true labels
true_labels = test_generator.classes

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(true_labels, predicted_labels)

# Calculate Accuracy
accuracy = accuracy_score(true_labels, predicted_labels)

# Calculate Precision
precision = precision_score(true_labels, predicted_labels)

# Calculate Recall
recall = recall_score(true_labels, predicted_labels)

print(f'MSE:       {mse:.5f}')
print(f'Accuracy:  {accuracy:.5f}')
print(f'Precision: {precision:.5f}')
print(f'Recall:    {recall:.5f}')
```

    16/16 [==============================] - 6s 363ms/step
    MSE:       0.19800
    Accuracy:  0.80200
    Precision: 0.91598
    Recall:    0.66500
    

### Classification Report


```python
# Generate classification report
report = classification_report(true_labels, predicted_labels, target_names=["Benign", "Malignant"])
print("Classification Report:\n", report)
```

    Classification Report:
                   precision    recall  f1-score   support
    
          Benign       0.74      0.94      0.83      1000
       Malignant       0.92      0.67      0.77      1000
    
        accuracy                           0.80      2000
       macro avg       0.83      0.80      0.80      2000
    weighted avg       0.83      0.80      0.80      2000
    
    

### Confution Matrix


```python
# Generate confusion matrix
cm = confusion_matrix(true_labels, predicted_labels)

# Display the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Benign", "Malignant"])
disp.plot(cmap='Blues', values_format='d')
plt.title("Confusion Matrix")
plt.show()
```


    
![png](pngs/output_31_0.png)
    


## Logistic Regression

### Creation


```python
def create_logistic_regression_model(input_shape, num_classes):
    """
    Create a logistic regression model.

    Parameters:
    - input_shape: Tuple, shape of the input data (e.g., (height, width, channels)).
    - num_classes: Integer, number of classes for classification.

    Returns:
    - lr_model: Compiled logistic regression model.
    """
    lr_model = Sequential()
  
    # Add an input layer with the specified input shape
    lr_model.add(tf.keras.Input(shape=input_shape))

    # Flatten the input
    lr_model.add(tf.keras.layers.Flatten())

    # Add a dense layer with the number of classes
    lr_model.add(tf.keras.layers.Dense(num_classes))
        
    # Apply softmax activation to the output layer
    lr_model.add(tf.keras.layers.Softmax())
    
    # Compile the model with Adam optimizer, sparse categorical crossentropy loss, and accuracy metric
    lr_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
    )

    return lr_model
```


```python
# Model parameters
input_shape = (img_width, img_height, 3)
num_classes = 2
```


```python
# Get model
lr_model = create_logistic_regression_model(input_shape, num_classes)

# Display the model architecture
lr_model.summary()
```

    Model: "sequential_8"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     flatten_7 (Flatten)         (None, 37632)             0         
                                                                     
     dense_16 (Dense)            (None, 2)                 75266     
                                                                     
     softmax_1 (Softmax)         (None, 2)                 0         
                                                                     
    =================================================================
    Total params: 75266 (294.01 KB)
    Trainable params: 75266 (294.01 KB)
    Non-trainable params: 0 (0.00 Byte)
    _________________________________________________________________
    

### Training and evaluating


```python
# Train
lr_history = lr_model.fit(
  train_generator,
  epochs=epochs,
  validation_data=test_generator  
)

# Evaluate
test_loss, test_acc = lr_model.evaluate(test_generator)
print('Test accuracy:', test_acc)
```

    Epoch 1/15
    93/93 [==============================] - 39s 420ms/step - loss: 0.6231 - accuracy: 0.7078 - val_loss: 0.6343 - val_accuracy: 0.6775
    Epoch 2/15
    93/93 [==============================] - 39s 415ms/step - loss: 0.6243 - accuracy: 0.7203 - val_loss: 0.8821 - val_accuracy: 0.6030
    Epoch 3/15
    93/93 [==============================] - 40s 426ms/step - loss: 0.6533 - accuracy: 0.7195 - val_loss: 0.4682 - val_accuracy: 0.7860
    Epoch 4/15
    93/93 [==============================] - 65s 697ms/step - loss: 0.7014 - accuracy: 0.7110 - val_loss: 0.5071 - val_accuracy: 0.7595
    Epoch 5/15
    93/93 [==============================] - 45s 480ms/step - loss: 0.5439 - accuracy: 0.7690 - val_loss: 0.6477 - val_accuracy: 0.6700
    Epoch 6/15
    93/93 [==============================] - 39s 421ms/step - loss: 0.6238 - accuracy: 0.7380 - val_loss: 0.8533 - val_accuracy: 0.6290
    Epoch 7/15
    93/93 [==============================] - 39s 415ms/step - loss: 0.6753 - accuracy: 0.7304 - val_loss: 0.7966 - val_accuracy: 0.6435
    Epoch 8/15
    93/93 [==============================] - 40s 432ms/step - loss: 0.5397 - accuracy: 0.7657 - val_loss: 0.4699 - val_accuracy: 0.7660
    Epoch 9/15
    93/93 [==============================] - 39s 420ms/step - loss: 0.6715 - accuracy: 0.7361 - val_loss: 1.1770 - val_accuracy: 0.5930
    Epoch 10/15
    93/93 [==============================] - 39s 419ms/step - loss: 1.0448 - accuracy: 0.7065 - val_loss: 0.7782 - val_accuracy: 0.6780
    Epoch 11/15
    93/93 [==============================] - 38s 409ms/step - loss: 0.5499 - accuracy: 0.7756 - val_loss: 0.5345 - val_accuracy: 0.7330
    Epoch 12/15
    93/93 [==============================] - 44s 471ms/step - loss: 0.5197 - accuracy: 0.7810 - val_loss: 0.4747 - val_accuracy: 0.7910
    Epoch 13/15
    93/93 [==============================] - 45s 484ms/step - loss: 0.6370 - accuracy: 0.7522 - val_loss: 0.6864 - val_accuracy: 0.6860
    Epoch 14/15
    93/93 [==============================] - 42s 452ms/step - loss: 0.6720 - accuracy: 0.7422 - val_loss: 0.5297 - val_accuracy: 0.7430
    Epoch 15/15
    93/93 [==============================] - 44s 474ms/step - loss: 0.7086 - accuracy: 0.7467 - val_loss: 0.7234 - val_accuracy: 0.6875
    16/16 [==============================] - 6s 394ms/step - loss: 0.7234 - accuracy: 0.6875
    Test accuracy: 0.6875
    

### Training Results


```python
# Plot training and test accuracy values
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(np.arange(1, len(lr_history.history['val_accuracy']) + 1), lr_history.history['val_accuracy'], label='Test Accuracy', color='darkgoldenrod')
plt.plot(np.arange(1, len(lr_history.history['accuracy']) + 1), lr_history.history['accuracy'], label='Training Accuracy', color='gold')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Test Accuracy')
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(range(1, len(lr_history.history['val_accuracy']) + 1))
plt.legend()

# Plot training and test loss values
plt.subplot(1, 2, 2)
plt.plot(np.arange(1, len(lr_history.history['val_loss']) + 1), lr_history.history['val_loss'], label='Test Loss', color='darkgoldenrod')
plt.plot(np.arange(1, len(lr_history.history['loss']) + 1), lr_history.history['loss'], label='Training Loss', color='gold')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Test Loss')
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(range(1, len(lr_history.history['val_loss']) + 1))
plt.legend()

plt.tight_layout()
plt.show()
```


    
![png](pngs/output_40_0.png)
    


### Model Evaluation Metrics


```python
# Evaluate the model on the test data and get predictions
predicted_probabilities = lr_model.predict(test_generator)

# Convert probabilities to binary predictions (0 or 1)
predicted_labels = np.round(predicted_probabilities).astype(np.int32)[:, 1]  # [:, 0] is the probabily to mistake.

# Get true labels
true_labels = test_generator.classes

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(true_labels, predicted_labels)

# Calculate Accuracy
accuracy = accuracy_score(true_labels, predicted_labels)

# Calculate Precision
precision = precision_score(true_labels, predicted_labels)

# Calculate Recall
recall = recall_score(true_labels, predicted_labels)

print(f'MSE:       {mse:.5f}')
print(f'Accuracy:  {accuracy:.5f}')
print(f'Precision: {precision:.5f}')
print(f'Recall:    {recall:.5f}')
```

    16/16 [==============================] - 7s 402ms/step
    MSE:       0.31250
    Accuracy:  0.68750
    Precision: 0.61935
    Recall:    0.97300
    

### Classification Report


```python
# Generate classification report
report = classification_report(true_labels, predicted_labels, target_names=["Benign", "Malignant"])
print("Classification Report:\n", report)
```

    Classification Report:
                   precision    recall  f1-score   support
    
          Benign       0.94      0.40      0.56      1000
       Malignant       0.62      0.97      0.76      1000
    
        accuracy                           0.69      2000
       macro avg       0.78      0.69      0.66      2000
    weighted avg       0.78      0.69      0.66      2000
    
    

### Confusion Matrix


```python
# Generate confusion matrix
cm = confusion_matrix(true_labels, predicted_labels)

# Display the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Benign", "Malignant"])
disp.plot(cmap='YlOrBr', values_format='d')
plt.title("Confusion Matrix")
plt.show()
```


    
![png](pngs/output_46_0.png)
    


## Convolutional Neural Network

### Creation


```python
def create_cnn_model(input_shape, num_classes, 
                    conv_layers=2, 
                    conv_filters=32,
                    conv_kernel_size=(3,3),
                    conv_activation='relu',
                    pool_size=(2,2),
                    learning_rate=0.0001):

    # Create sequential model
    cnn_model = Sequential()
    
    # Add input layer
    cnn_model.add(Conv2D(conv_filters, kernel_size=conv_kernel_size, activation=conv_activation, input_shape=input_shape))
    cnn_model.add(BatchNormalization())
    cnn_model.add(MaxPooling2D(pool_size=pool_size))

    # Add convolutional layers
    for i in range(conv_layers):
        cnn_model.add(Conv2D(conv_filters, 
                             kernel_size=conv_kernel_size, 
                             activation=conv_activation))
        cnn_model.add(BatchNormalization())
        cnn_model.add(MaxPooling2D(pool_size=pool_size))

    # Fully connected layer with dropout
    cnn_model.add(Flatten())
    cnn_model.add(BatchNormalization())
    cnn_model.add(Dense(num_classes, activation='sigmoid'))

    # Compile the model with specified learning rate
    optimizer = Adam(learning_rate=learning_rate)
    cnn_model.compile(
                    optimizer=optimizer,
                    loss='binary_crossentropy',
                    metrics=['accuracy'])
  
    return cnn_model
```


```python
# Model parameters
input_shape = (img_height, img_width, 3)
num_classes = 1
```


```python
# Get model
cnn_model = create_cnn_model(input_shape, num_classes)

# Display the model architecture
cnn_model.summary()
```

    Model: "sequential_11"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     conv2d_18 (Conv2D)          (None, 110, 110, 32)      896       
                                                                     
     batch_normalization_20 (Ba  (None, 110, 110, 32)      128       
     tchNormalization)                                               
                                                                     
     max_pooling2d_16 (MaxPooli  (None, 55, 55, 32)        0         
     ng2D)                                                           
                                                                     
     conv2d_19 (Conv2D)          (None, 53, 53, 32)        9248      
                                                                     
     batch_normalization_21 (Ba  (None, 53, 53, 32)        128       
     tchNormalization)                                               
                                                                     
     max_pooling2d_17 (MaxPooli  (None, 26, 26, 32)        0         
     ng2D)                                                           
                                                                     
     conv2d_20 (Conv2D)          (None, 24, 24, 32)        9248      
                                                                     
     batch_normalization_22 (Ba  (None, 24, 24, 32)        128       
     tchNormalization)                                               
                                                                     
     max_pooling2d_18 (MaxPooli  (None, 12, 12, 32)        0         
     ng2D)                                                           
                                                                     
     flatten_9 (Flatten)         (None, 4608)              0         
                                                                     
     batch_normalization_23 (Ba  (None, 4608)              18432     
     tchNormalization)                                               
                                                                     
     dense_18 (Dense)            (None, 1)                 4609      
                                                                     
    =================================================================
    Total params: 42817 (167.25 KB)
    Trainable params: 33409 (130.50 KB)
    Non-trainable params: 9408 (36.75 KB)
    _________________________________________________________________
    

### Training and evaluating


```python
# Train
cnn_history = cnn_model.fit(
  train_generator,
  epochs=epochs,
  validation_data=test_generator  
)

# Evaluate
test_loss, test_acc = cnn_model.evaluate(test_generator)
print('Test accuracy:', test_acc)
```

    Epoch 1/15
    93/93 [==============================] - 276s 3s/step - loss: 0.4747 - accuracy: 0.7728 - val_loss: 0.7628 - val_accuracy: 0.5000
    Epoch 2/15
    93/93 [==============================] - 250s 3s/step - loss: 0.3463 - accuracy: 0.8454 - val_loss: 0.8755 - val_accuracy: 0.5000
    Epoch 3/15
    93/93 [==============================] - 259s 3s/step - loss: 0.3156 - accuracy: 0.8637 - val_loss: 0.9882 - val_accuracy: 0.5010
    Epoch 4/15
    93/93 [==============================] - 215s 2s/step - loss: 0.2964 - accuracy: 0.8749 - val_loss: 0.7901 - val_accuracy: 0.5340
    Epoch 5/15
    93/93 [==============================] - 240s 3s/step - loss: 0.2813 - accuracy: 0.8816 - val_loss: 0.6494 - val_accuracy: 0.6055
    Epoch 6/15
    93/93 [==============================] - 269s 3s/step - loss: 0.2729 - accuracy: 0.8856 - val_loss: 0.5424 - val_accuracy: 0.6810
    Epoch 7/15
    93/93 [==============================] - 250s 3s/step - loss: 0.2627 - accuracy: 0.8870 - val_loss: 0.3982 - val_accuracy: 0.7845
    Epoch 8/15
    93/93 [==============================] - 262s 3s/step - loss: 0.2582 - accuracy: 0.8936 - val_loss: 0.2937 - val_accuracy: 0.8720
    Epoch 9/15
    93/93 [==============================] - 263s 3s/step - loss: 0.2508 - accuracy: 0.8949 - val_loss: 0.2495 - val_accuracy: 0.9050
    Epoch 10/15
    93/93 [==============================] - 268s 3s/step - loss: 0.2439 - accuracy: 0.8978 - val_loss: 0.2601 - val_accuracy: 0.9010
    Epoch 11/15
    93/93 [==============================] - 247s 3s/step - loss: 0.2394 - accuracy: 0.8987 - val_loss: 0.2707 - val_accuracy: 0.8945
    Epoch 12/15
    93/93 [==============================] - 249s 3s/step - loss: 0.2334 - accuracy: 0.9019 - val_loss: 0.2549 - val_accuracy: 0.9020
    Epoch 13/15
    93/93 [==============================] - 237s 3s/step - loss: 0.2267 - accuracy: 0.9070 - val_loss: 0.2567 - val_accuracy: 0.8920
    Epoch 14/15
    93/93 [==============================] - 215s 2s/step - loss: 0.2221 - accuracy: 0.9080 - val_loss: 0.2400 - val_accuracy: 0.9040
    Epoch 15/15
    93/93 [==============================] - 231s 2s/step - loss: 0.2168 - accuracy: 0.9113 - val_loss: 0.2845 - val_accuracy: 0.8775
    16/16 [==============================] - 13s 764ms/step - loss: 0.2845 - accuracy: 0.8775
    Test accuracy: 0.8774999976158142
    

### Training Results


```python
# Plot training and test accuracy values
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(np.arange(1, len(cnn_history.history['val_accuracy']) + 1), cnn_history.history['val_accuracy'], label='Test Accuracy', color='darkolivegreen')
plt.plot(np.arange(1, len(cnn_history.history['accuracy']) + 1), cnn_history.history['accuracy'], label='Training Accuracy', color='yellowgreen')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Test Accuracy')
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(range(1, len(cnn_history.history['val_accuracy']) + 1))
plt.legend()

# Plot training and test loss values
plt.subplot(1, 2, 2)
plt.plot(np.arange(1, len(cnn_history.history['val_loss']) + 1), cnn_history.history['val_loss'], label='Test Loss', color='darkolivegreen')
plt.plot(np.arange(1, len(cnn_history.history['loss']) + 1), cnn_history.history['loss'], label='Training Loss', color='yellowgreen')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Test Loss')
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(range(1, len(cnn_history.history['val_loss']) + 1))
plt.legend()

plt.tight_layout()
plt.show()
```


    
![png](pngs/output_55_0.png)
    


### Model Evaluation Metrics


```python
# Evaluate the model on the test data and get predictions
predicted_probabilities = cnn_model.predict(test_generator)

# Convert probabilities to binary predictions (0 or 1)
predicted_labels = np.round(predicted_probabilities).astype(np.int32)[:, 0]  # [:, 0] is the probabily to mistake.

# Get true labels
true_labels = test_generator.classes

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(true_labels, predicted_labels)

# Calculate Accuracy
accuracy = accuracy_score(true_labels, predicted_labels)

# Calculate Precision
precision = precision_score(true_labels, predicted_labels)

# Calculate Recall
recall = recall_score(true_labels, predicted_labels)

print(f'MSE:       {mse:.5f}')
print(f'Accuracy:  {accuracy:.5f}')
print(f'Precision: {precision:.5f}')
print(f'Recall:    {recall:.5f}')
```

    16/16 [==============================] - 15s 920ms/step
    MSE:       0.12250
    Accuracy:  0.87750
    Precision: 0.91257
    Recall:    0.83500
    

### Classification Report


```python
# Generate classification report
report = classification_report(true_labels, predicted_labels, target_names=["Benign", "Malignant"])
print("Classification Report:\n", report)
```

    Classification Report:
                   precision    recall  f1-score   support
    
          Benign       0.85      0.92      0.88      1000
       Malignant       0.91      0.83      0.87      1000
    
        accuracy                           0.88      2000
       macro avg       0.88      0.88      0.88      2000
    weighted avg       0.88      0.88      0.88      2000
    
    

### Confusion Matrix


```python
# Generate confusion matrix
cm = confusion_matrix(true_labels, predicted_labels)

# Display the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Benign", "Malignant"])
disp.plot(cmap='Greens', values_format='d')
plt.title("Confusion Matrix")
plt.show()
```


    
![png](pngs/output_61_0.png)
    


## Comparation


```python
# Create a figure with 4 subplots
plt.figure(figsize=(18, 12))

# Comparison of Training Accuracy for all models
plt.subplot(2, 2, 1)
plt.plot(np.arange(1, len(history.history['accuracy']) + 1), history.history['accuracy'], label='NN', color="skyblue")
plt.plot(np.arange(1, len(lr_history.history['accuracy']) + 1), lr_history.history['accuracy'], label='Logistic Regression', color='gold')
plt.plot(np.arange(1, len(cnn_history.history['accuracy']) + 1), cnn_history.history['accuracy'], label='CNN', color='yellowgreen')
plt.xlabel('Epoch')
plt.ylabel('Training Accuracy')
plt.title('Training Accuracy Comparison')
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(range(1, len(history.history['accuracy']) + 1))
plt.legend()

# Comparison of Training Loss for all models
plt.subplot(2, 2, 2)
plt.plot(np.arange(1, len(history.history['loss']) + 1), history.history['loss'], label='NN', color="skyblue")
plt.plot(np.arange(1, len(lr_history.history['loss']) + 1), lr_history.history['loss'], label='Logistic Regression', color='gold')
plt.plot(np.arange(1, len(cnn_history.history['loss']) + 1), cnn_history.history['loss'], label='CNN', color='yellowgreen')
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.title('Training Loss Comparison')
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(range(1, len(history.history['loss']) + 1))
plt.legend()

# Comparison of test Accuracy for all models
plt.subplot(2, 2, 3)
plt.plot(np.arange(1, len(history.history['val_accuracy']) + 1), history.history['val_accuracy'], label='NN', color="steelblue")
plt.plot(np.arange(1, len(lr_history.history['val_accuracy']) + 1), lr_history.history['val_accuracy'], label='Logistic Regression', color='darkgoldenrod')
plt.plot(np.arange(1, len(cnn_history.history['val_accuracy']) + 1), cnn_history.history['val_accuracy'], label='CNN', color='darkolivegreen')
plt.xlabel('Epoch')
plt.ylabel('Test Accuracy')
plt.title('Test Accuracy Comparison')
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(range(1, len(history.history['val_accuracy']) + 1))
plt.legend()

# Comparison of Test Loss for all models
plt.subplot(2, 2, 4)
plt.plot(np.arange(1, len(history.history['val_loss']) + 1), history.history['val_loss'], label='NN', color="steelblue")
plt.plot(np.arange(1, len(lr_history.history['val_loss']) + 1), lr_history.history['val_loss'], label='Logistic Regression', color='darkgoldenrod')
plt.plot(np.arange(1, len(cnn_history.history['val_loss']) + 1), cnn_history.history['val_loss'], label='CNN', color='darkolivegreen')
plt.xlabel('Epoch')
plt.ylabel('Test Loss')
plt.title('Test Loss Comparison')
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(range(1, len(history.history['val_loss']) + 1))
plt.legend()

# Adjust layout for better visualization
plt.tight_layout()

# Show the combined plot
plt.show()
```


    
![png](pngs/output_63_0.png)
    


# Report

## Prediction Results


```python
# Get a batch of images and labels from the test generator
batch_images, batch_labels = test_generator.next()

# Select 6 random indices from the batch
random_indices = np.random.choice(len(batch_labels), 6, replace=False)

# Create a figure with 2 rows and 3 columns
plt.figure(figsize=(14, 8))

# Display images with predicted and true labels
for i, index in enumerate(random_indices, start=1):
    plt.subplot(2, 3, i)
    plt.imshow(batch_images[index])
    plt.axis('off')
    
    # Determine the predicted class based on a threshold (e.g., 0.5)
    predicted_class = 1 if predicted_probabilities[index][0] >= 0.5 else 0
    
    # Check if the prediction is correct
    is_correct = predicted_class == batch_labels[index]
    
    # Use checkmark (✔) for correct and cross (✘) for incorrect
    sign = "✔" if is_correct else "✘"
    
    # Display prediction probability, predicted class, and true class
    plt.title(f"Prediction: {'Malignant' if predicted_class == 1 else 'Benign'}\nTrue: {'Malignant' if batch_labels[index] == 1 else 'Benign'}\n{predicted_probabilities[index][0]:.2f}\n{sign}")

# Adjust layout for better visualization
plt.tight_layout()
plt.show()
```


    
![png](pngs/output_66_0.png)
    

