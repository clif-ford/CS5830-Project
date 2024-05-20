# Data Preprocessing Task

<b>Objective</b> - Download The EMNIST by class dataset. Preprocess it and save it in a csv file for later use. <br>

<b>Dataset</b> - https://www.nist.gov/srd/nist-special-database-19 

## Requirements

- docker
- docker-compose

Since docker is used, No other requirements are needed.

## How to initialize
Run docker compose airflow db init service initially to set up airflow. Once the service has exited, docker compose up. The Airflow GUI will be available to use at localhost:8080. Simply run the DAG to start the process. 

Airflow will initially download the initial database in zip form and store it in `files/downloaded_files`. 

Once this task is done, airflow will proceed to unzip and save the extracted files in `files/extracted_files`

Finally, airflow will use the extracted_files to create a `final_output.csv` file in `files/preprocessed_files`.

## CSV structure

The CSV is a `112800 x 785` table. Each row corresponds to 1 entry. The first 784 columns are the normalised pixel values of each of the image's cells. The last column is the label. 

There are a total of 62 classes corresponding to Capital letters + small letters + numbers. However, we use only 26 mixed assortment of letters. Each letter has 2400 entries. Hence the dataset we used has dimensions `62400 x 785`.
