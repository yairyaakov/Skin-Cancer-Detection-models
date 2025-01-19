#!/usr/bin/env python
# coding: utf-8

# # Skin Cancer Detection models 🩺 | BaseLine Model, CNN, NN and Logistic Regression

# #### Prologue
# This notebook explores melanoma classification using machine learning.
# The objective is **construct** and **analyze** four models: baseLine Model, neural network, logistic regression, and a CNN.
# 
# #### Dataset Overview
# Comprising 13,900 uniformly-sized images at 224 x 224 pixels, which provides a comprehensive portrayal of diverse manifestations of melanoma. Each image is meticulously labeled as either `benign` or `malignant`.

# In[1]:


import asyncio
asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
#----------
import os
import random
#----------
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
from tensorflow.compat.v1 import reset_default_graph

#----------
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report, mean_squared_error, accuracy_score, precision_score, recall_score
#----------
import pandas as pd
#----------
import matplotlib.pyplot as plt
#----------
import numpy as np
#----------
from PIL import Image
#----------
import warnings
warnings.filterwarnings('ignore')
#----------
import seaborn as sns
#----------
from IPython.display import Image


# ## Load and preprocess the dataset

# In[2]:


# Path to the dataset archive
archive_path = r'C:\Users\yair8\Skin-Cancer-Detection-Project\dataSet'


# In[3]:


# Define the main folder path after extraction
main_folder_path = os.path.splitext(archive_path)[0]  # Remove the extension


# The main folder contains 2 folders - train and test - and each of them conatins 2 folder - Benign and Malignant

# In[4]:


# Define subfolders
data_folders = ["train", "test"]
class_folders = ["Benign", "Malignant"]


# In[5]:


# Paths for train and test data
train_data_path = os.path.join(main_folder_path, data_folders[0])
test_data_path = os.path.join(main_folder_path, data_folders[1])

print("Train Data Path:", train_data_path)
print("Test Data Path:", test_data_path)

print("Train folder content:")
for item in os.listdir(train_data_path):
    print(f"- {item}")

print("Test folder content:")
for item in os.listdir(test_data_path):
    print(f"- {item}")


# Set hyperparamaters 

# In[6]:


img_width, img_height = 112, 112 
batch_size = 32
epochs = 15    


# In[7]:


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


# In[8]:


def plotCount(generator):
    # ספירת כמות התמונות בקטגוריות Benign ו-Malignant
    malignant_count = sum(generator.labels == 1)
    benign_count = sum(generator.labels == 0)
    total_count = len(generator.labels)

    print(f"Total images is {total_count}")
    print(f"Total malignant images is {malignant_count} ({round(malignant_count / total_count * 100, 2)}%)")
    print(f"Total benign images is {benign_count} ({round(benign_count / total_count * 100, 2)}%)")

    sns.set_style("whitegrid")
    plt.figure(figsize=(6, 4))
    sns.countplot(x=generator.labels)
    plt.title("Image Count for Benign vs Malignant")
    plt.xlabel("Label (0=Benign, 1=Malignant)")
    plt.ylabel("Count")
    plt.show()


# In[9]:


plotCount(train_generator)


# In[10]:


plotCount(test_generator)


# # Models

# This section involves constructing three models: a Neural Network (NN), Logistic Regression, and Convolutional Neural Network (CNN). Each model is analyzed individually, followed by a comparative evaluation to discern their respective performance characteristics.

# ## BaseLine Model

# ### Creation

# In[11]:


def create_baseline_model(train_generator):
      # Extract all labels from the entire dataset (train_generator.classes)
    labels = train_generator.classes

    # Find the most frequent class (numeric index)
    majority_class_index = np.argmax(np.bincount(labels))

    for class_name, index in train_generator.class_indices.items():
        if index == majority_class_index:
            print(f"The majority class is: {class_name} (Index: {majority_class_index})")
            break

    return majority_class_index


# ### Training and Evaluating

# In[12]:


def evaluate_baseline_model(majority_class_index, test_generator):

    true_labels = test_generator.classes

    predicted_labels = np.full_like(true_labels, fill_value=majority_class_index)

    return predicted_labels, true_labels


# ### Training Results

# In[13]:


# Create the baseline model
baseline_majority_class = create_baseline_model(train_generator)

# Evaluate the baseline model
baseline_predictions, baseline_true_labels = evaluate_baseline_model(baseline_majority_class, test_generator)

# Display results
print(f"Baseline Predictions (first 1001): {baseline_predictions[:1001]}")
print(f"True Labels (first 1001): {baseline_true_labels[:1001]}")


# ### Model Evaluation Metrics

# In[14]:


def evaluate_model_performance(true_labels, predicted_labels, pos_label=1):

    mse = mean_squared_error(true_labels, predicted_labels)

    accuracy = accuracy_score(true_labels, predicted_labels) * 100

    precision = precision_score(true_labels, predicted_labels, pos_label=pos_label) * 100

    recall = recall_score(true_labels, predicted_labels, pos_label=pos_label) * 100

    print(f"Mean Squared Error: {mse:.2f}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Precision: {precision:.2f}%")
    print(f"Recall: {recall:.2f}%")

    return accuracy, precision, recall, mse


# In[15]:


evaluate_model_performance(baseline_true_labels, baseline_predictions)


# ### Classification Report

# In[16]:


# Generate classification report
baseline_report = classification_report(
    baseline_true_labels,
    baseline_predictions,
    target_names=list(test_generator.class_indices.keys())
)

# Display the classification report
print("\nClassification Report:\n")
print(baseline_report)

# Accuracy = Number of correct predictions/Total number of samples
# Precision = True Positives*(True Positives+False Positives)  
# Recall = True Positives*(True Positives+False Negatives) 
# F1 Score = ((Precision⋅Recall)*2)/(Precision+Recall) 
# Support = the number of true examples from each class 
# Macro avg = a simple average of all the indices between the classes.
# Weighted avg = the weighted average of all the indices between the classes, considering the number of true examples from each class.


# ### Confusion Matrix

# In[17]:


baseline_cm = confusion_matrix(baseline_true_labels, baseline_predictions)

disp = ConfusionMatrixDisplay(confusion_matrix=baseline_cm, display_labels=["Benign", "Malignant"])

fig, ax = plt.subplots(figsize=(6, 6))
disp.plot(cmap='Blues', values_format='d', ax=ax)
ax.grid(False)
plt.title("Confusion Matrix for Baseline Model", fontsize=14)
plt.xlabel("Predicted Label", fontsize=12)
plt.ylabel("True Label", fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[18]:


reset_default_graph()


# ## Neural Network 

# ### Creation

# In[19]:


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
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy'])

    return model


# In[20]:


# Define NN sizes
hidden_layer_sizes = [128, 64, 64]
num_hidden_layers = len(hidden_layer_sizes)

# Get the NN model
nn_model = create_nn(num_hidden_layers, hidden_layer_sizes)

# Display the model architecture
nn_model.summary()

# Output Shape: specifies the size of the output for each layer.
# Param:specifies the number of parameters (weights and biases) learned for each layer.
# (The number of neurons in the previous layer) * (The number of neurons in the current layer) + (The number of biases in the current layer)
# For example: The first layer has 128*112*112*3 + 128 = 4,817,024 parameters (+128 biases)


# ### Training and evaluating

# In[21]:


# Define the callback to use TensorBoard
tensorboard_callback = TensorBoard(log_dir='./logs', histogram_freq=1)

# Train the model with TensorBoard callback
history = nn_model.fit(train_generator, epochs=epochs, validation_data=test_generator, callbacks=[tensorboard_callback])


# In[22]:


last_val_accuracy = history.history['val_accuracy'][-1]

print(f"Test Accuracy: {last_val_accuracy}")


# ### Training Results

# In[23]:


def plot_history(history, name, metric):

    label_val = 'val_%s' % metric

    # Extract the training and validation metrics
    train = history.history[metric]
    test = history.history[label_val]

    # Create count of the number of epochs
    epoch_count = range(1, len(train) + 1)

    # Plot the training and validation metric
    plt.plot(epoch_count, train, 'r-', label='Train')
    plt.plot(epoch_count, test, 'b--', label='Test')

    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel(metric)
    plt.title(f'{metric.capitalize()} History for {name}')

    plt.show()


# In[24]:


plot_history(history, 'NN', 'accuracy')


# In[25]:


plot_history(history, 'NN', 'loss')


# ### Model Evaluation Metrics

# In[26]:


predicted_probabilities = nn_model.predict(test_generator)

# Convert probabilities to binary predictions (0 or 1)
predicted_labels = np.where(predicted_probabilities > 0.5, 1, 0).astype(np.int32)

# Get true labels
true_labels = test_generator.classes

# Use the evaluate_model_performance function
accuracy, precision, recall, mse = evaluate_model_performance(true_labels, predicted_labels)


# ### Classification Report

# In[27]:


# Generate classification report
report = classification_report(true_labels, predicted_labels, target_names=["Benign", "Malignant"])
print("Classification Report:\n", report)


# ### Confution Matrix

# In[28]:


nn_cm = confusion_matrix(true_labels, predicted_labels)

disp = ConfusionMatrixDisplay(confusion_matrix=nn_cm, display_labels=["Benign", "Malignant"])

fig, ax = plt.subplots(figsize=(6, 6))
disp.plot(cmap='Blues', values_format='d', ax=ax)
ax.grid(False)
plt.title("Confusion Matrix for Neural Network Model", fontsize=14)
plt.xlabel("Predicted Label", fontsize=12)
plt.ylabel("True Label", fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[29]:


K.clear_session()


# ## Logistic Regression

# ### Creation

# In[30]:


def create_logistic_regression_model(input_shape):

    model = Sequential()

    model.add(tf.keras.Input(shape=input_shape))

    model.add(Flatten())

    model.add(Dense(1))

    model.add(tf.keras.layers.Activation('sigmoid'))

    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model


# In[31]:


# Model parameters
input_shape = (img_width, img_height, 3)

# Get model
logistic_model = create_logistic_regression_model(input_shape)

# Display the model architecture
logistic_model.summary()


# ### Training and evaluating

# In[32]:


tensorboard_callback = TensorBoard(log_dir='./logs', histogram_freq=1)

# Train the model with TensorBoard callback
history = logistic_model.fit(train_generator, epochs=epochs, validation_data=test_generator, callbacks=[tensorboard_callback])


# In[33]:


last_val_accuracy = history.history['val_accuracy'][-1]

print(f"Test Accuracy: {last_val_accuracy}")


# ### Training Results

# In[34]:


plot_history(history, 'logistic', 'accuracy')


# In[35]:


plot_history(history, 'logistic', 'loss')


# ### Model Evaluation Metrics

# In[36]:


predicted_probabilities = logistic_model.predict(test_generator)

# Convert probabilities to binary predictions (0 or 1)
predicted_labels = np.where(predicted_probabilities > 0.5, 1, 0).astype(np.int32)

# Get true labels
true_labels = test_generator.classes

# Use the evaluate_model_performance function
accuracy, precision, recall, mse = evaluate_model_performance(true_labels, predicted_labels)


# ### Classification Report

# In[37]:


# Generate classification report
report = classification_report(true_labels, predicted_labels, target_names=["Benign", "Malignant"])
print("Classification Report:\n", report)


# ### Confusion Matrix

# In[38]:


logistic_cm = confusion_matrix(true_labels, predicted_labels)

disp = ConfusionMatrixDisplay(confusion_matrix=logistic_cm, display_labels=["Benign", "Malignant"])

fig, ax = plt.subplots(figsize=(6, 6))
disp.plot(cmap='Blues', values_format='d', ax=ax)
ax.grid(False)
plt.title("Confusion Matrix for Logistic Regression Model", fontsize=14)
plt.xlabel("Predicted Label", fontsize=12)
plt.ylabel("True Label", fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[39]:


K.clear_session()


# ## Convolutional Neural Network

# ### Creation

# In[40]:


# יצירת המודל
def Convolutional_neural_network_model(input_shape):
    model = Sequential()  # מודל סיקוונסיאלי, בו כל השכבות ממוקמות אחת אחרי השנייה

    # שכבת קונבולוציה ראשונה
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', strides=1, input_shape=input_shape))
    # Conv2D: שכבת קונבולוציה עם 64 פילטרים בגודל 3x3
    # activation='relu': הפונקציה הסיגמואידית relu היא פונקציית הפעלה לא לינארית
    # padding='same': כלומר הפלט יהיה באותו גודל כמו הקלט (הוספת פדינג סביב התמונה)
    # strides=1: הסטרייד (הזזה) הוא 1, כלומר נעבור על כל פיקסל בתמונה
    # input_shape: נתוני הקלט, מימדי התמונה (רוחב, גובה, ערוצים)
    model.add(BatchNormalization())  # BatchNormalization: נרמול הפלט של השכבה כדי יהיה בעל ממוצע 0 ושונות קבועה להגברת היציבות
    model.add(MaxPooling2D((2, 2), strides=2))  # MaxPooling2D: פוולינג מקסימלי בגודל 2x2 עם סטרייד של 2

    # שכבת קונבולוציה שניה
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', strides=1))
    # Conv2D: שכבת קונבולוציה עם 128 פילטרים בגודל 3x3
    model.add(BatchNormalization())  # BatchNormalization: נרמול פנימי של השכבה
    model.add(MaxPooling2D((2, 2), strides=2))  # MaxPooling2D: פוולינג מקסימלי בגודל 2x2 עם סטרייד של 2

    # שכבת קונבולוציה שלישית
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', strides=1))
    # Conv2D: שכבת קונבולוציה עם 256 פילטרים בגודל 3x3
    model.add(BatchNormalization())  # BatchNormalization: נרמול פנימי של השכבה
    model.add(MaxPooling2D((2, 2), strides=2))  # MaxPooling2D: פוולינג מקסימלי בגודל 2x2 עם סטרייד של 2

    # Flatten - הופכים את הפלט לווקטור שטוח (לכדי לחבר אותו לשכבות דנס)
    model.add(Flatten())
    # Flatten: הפיכת מטריצה לתשדורת אחת שטוחה (וקטור)

    # שכבת Dense עם רגוליזציה L2
    model.add(Dense(256, activation='relu', kernel_regularizer=l2(0.001)))
    # Dense: שכבת Fully Connected עם 512 נוירונים, פונקציית הפעלה relu
    # kernel_regularizer=l2(0.001): רגוליזציה L2 על מנת למנוע אוברפיטינג (הוספת קנס לכמות האיברים בשכבה)
    model.add(BatchNormalization())  # BatchNormalization: נרמול פנימי של השכבה
    model.add(Dropout(0.5))  # Dropout: השמטת חצי מהנוירונים בשכבה לשיפור הכללה ולמניעת אוברפיטינג

    # שכבת פלט עם פונקציית הפעלה Sigmoid עבור סיווג בינארי
    model.add(Dense(1, activation='sigmoid'))
    # Dense: שכבת פלט עם נוירון אחד
    # activation='sigmoid': פונקציית הפעלה סיגמואיד המתאימה לסיווג בינארי (הפלט בין 0 ל-1)

    # קומפילציה של המודל
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
    # Adam: אופטימיזטור Adam עם קצב למידה נמוך של 0.0001
    # loss='binary_crossentropy': פונקציית הפסד לסיווג בינארי
    # metrics=['accuracy']: מדד הביצועים יהיה הדיוק

    # הגדרת הפסקה מוקדמת (EarlyStopping)
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    # monitor='val_loss': המעקב יהיה אחרי מדד האובדן (loss)
    # patience=5: אם לא תהיה שיפור במשך 5 אפוקים, תבוצע הפסקה מוקדמת
    # restore_best_weights=True: תחזור למשקולות הטובות ביותר שהיו עד כה

    return model  # מחזירים את המודל שהוגדר


# In[41]:


# Model parameters
input_shape = (112, 112, 3)

# Get model
cnn_model = Convolutional_neural_network_model(input_shape)

# Display the model architecture
cnn_model.summary()


# ### Training and evaluating

# In[42]:


tensorboard_callback = TensorBoard(log_dir='./logs', histogram_freq=1)

# Train the model with TensorBoard callback
history = cnn_model.fit(train_generator, epochs=epochs, validation_data=test_generator, callbacks=[tensorboard_callback])


# In[43]:


last_val_accuracy = history.history['val_accuracy'][-1]

print(f"Test Accuracy: {last_val_accuracy}")


# ### Training Results

# In[44]:


plot_history(history, 'cnn', 'accuracy')


# In[45]:


plot_history(history, 'cnn', 'loss')


# ### Model Evaluation Metrics

# In[46]:


predicted_probabilities = cnn_model.predict(test_generator)

# Convert probabilities to binary predictions (0 or 1)
predicted_labels = np.where(predicted_probabilities > 0.5, 1, 0).astype(np.int32)

# Get true labels
true_labels = test_generator.classes

# Use the evaluate_model_performance function
accuracy, precision, recall, mse = evaluate_model_performance(true_labels, predicted_labels)


# ### Classification Report

# In[47]:


report = classification_report(true_labels, predicted_labels, target_names=["Benign", "Malignant"])
print("Classification Report:\n", report)


# ### Confusion Matrix

# In[48]:


cnn_cm = confusion_matrix(true_labels, predicted_labels)

disp = ConfusionMatrixDisplay(confusion_matrix=cnn_cm, display_labels=["Benign", "Malignant"])

fig, ax = plt.subplots(figsize=(6, 6))
disp.plot(cmap='Blues', values_format='d', ax=ax)
ax.grid(False)
plt.title("Confusion Matrix for CNN Model", fontsize=14)
plt.xlabel("Predicted Label", fontsize=12)
plt.ylabel("True Label", fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# # Report

# ## Prediction Results

# In[49]:


num_images = 20
images, labels = next(test_generator)
predictions = cnn_model.predict(images)
cols = 5
rows = num_images // cols + (num_images % cols > 0)

plt.figure(figsize=(15, 5 * rows))
for i in range(num_images):
    plt.subplot(rows, cols, i + 1)
    plt.imshow(images[i])
    plt.axis('off')

    true_label = "malignant" if labels[i] == 1 else "benign"
    predicted_label = "malignant" if predictions[i] > 0.5 else "benign"

    color = "green" if true_label == predicted_label else "red"

    plt.title(f"True: {true_label}\nPred: {predicted_label}", color=color, fontsize=10)

plt.tight_layout()
plt.show()

