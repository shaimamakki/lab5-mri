# Lab 5  

### A simple explanation of everything I did from start to end


---

# 

# Introduction 
The main idea was to take raw MRI images, clean them, extract useful information from them, train a machine learning model, and finally register the model in Azure ML.
BUT, I tried creating notebooks and features extraction using Azure ML and could not do it because it kep asking for some permissions and stuff. I even created my compute instance and cluster still couldnt work. So, rather I did everything using Databricks.I tried to keep the whole project clear and organized, moving through the Bronze → Silver → Gold workflow. A proper lakehouse structure.
**Summary of what I did in Lab 5:**

**• Opened Azure ML Studio and tried to follow the official pipeline steps for the MRI project.**  
**• I attempted to create a Feature Set for the Silver layer, but the “Create” button was disabled.**  
**• Azure ML showed that Feature Sets can only be created using SDK/CLI, and I did not have the permissions needed.**  
**• Because of this, I could not finish the Feature Store part inside Azure ML.**

**• I tried using the Azure ML terminal to run commands, but the terminal option was also not available in my workspace.**  
**• I also checked different areas of Azure ML Studio, but everything related to Feature Store creation was locked.**

**• Since the Silver feature extraction pipeline could not be completed in Azure ML, I switched to Databricks.**  
**• I used my existing Goodreads Databricks environment because it already had a working cluster and permissions.**  
**• I loaded my MRI images and ran my feature extraction using Databricks notebooks instead of Azure ML.**  
**• The main reason: Azure ML kept blocking the steps due to missing permissions, while Databricks worked without issues, and i did using pySpark.**

**Bronze Layer (Raw Data): (01_bronze_create_raw_table.ipynb)**
- first i ingested the images provided to my storage account using terminal.
- then, I stored the raw MRI dataset (tumor / no_tumor folders) in the Bronze layer of my storage account.
- I created a raw table/view so the images could be easily accessed in Databricks.**
- I didn’t do any cleaning here — just stored the raw data.
- I checked that the folder structure was correct and could be accessed by both Azure ML and Databricks.  
- I confirmed that the raw files were readable and ready for feature extraction. 
- No transformations were done here — this layer only kept the original MRI images.

**Silver Layer (02_silver_feature_extraction.ipynb):**
- I first tried to do the Silver layer in Azure ML, but I couldn’t create a Feature Set because of missing permissions.
- I switched to Databricks and completed the full Silver layer there.
- I loaded the Bronze images and extracted features (like GLCM and basic texture/statistical features).
- I cleaned and organized the extracted features into a proper Silver dataset.
- I saved the Silver dataset back to storage so I could use it in the Gold layer.

**Gold Layer — Component A (03_gold_component_A_feature_retrieval.ipynb):**
- I loaded the Silver features into my notebook.
- I checked for missing values, shapes, and basic feature quality.
- I prepared the dataset so I could move to feature selection and modeling.

**Gold Layer — Component B (04_gold_component_B_feature_selection.ipynb):**
- I explored the Silver features and removed any irrelevant or constant features.
- I selected the most useful features for tumor classification.
- I finalized the feature set that would be used for training the models.

**Gold Layer — Component C (05_gold_component_C_train_eval.ipynb):**
- I split the data into train and test sets.**
- I trained multiple models: Logistic Regression, Random Forest, and XGBoost.**
- I compared their performance and accuracy.**
- XGBoost gave me the best accuracy overall.**
- I saved the best model as my final Gold-layer output.**

**• In the Gold layer, I trained and tested models on the extracted features.**  
- Model I tried: **RandomForestClassifier**  
**• Accuracy results:**  
 'test_accuracy': 0.6862745098039216,
 'num_features_used': 34,
 'training_time_seconds': 5.9052510261535645
I got a test accuracy of about 68.63% (0.6863).
I used 34 selected features for training.
My model took around 5.9 seconds to train.
I evaluated the predictions and saved the final model as my Gold-layer output.

After training, I evaluated the model results, compared them, and saved the final model output as part of the Gold layer. 
I planned to register the model in Azure ML, but registration also required permissions I did not have.  
Because of that, I used Databricks outputs instead of Azure ML model registry.










