#importing required libraries
from airflow import DAG
from airflow.models import Variable
from airflow.operators.python_operator import PythonOperator

from zipfile import ZipFile, ZIP_DEFLATED
from urllib.request import urlretrieve
from pathlib import Path

from datetime import datetime
import os
import re
import numpy as np
from PIL import Image
from zipfile import ZipFile
import pandas as pd

#Variables
data_url = 'https://s3.amazonaws.com/nist-srd/SD19/by_class.zip'
download_path = '/opt/airflow/files/Downloaded_files/by_class.zip'

#Dictionary to help convert ASCII to English
ASCII_dict = {
    "41" : "A",
    "61" : "a",
    "42" : "B",
    "62" : "b",
    "43" : "C",
    "63" : "c",
    "44" : "D",
    "64" : "d",
    "45" : "E",
    "65" : "e",
    "46" : "F",
    "66" : "f",
    "47" : "G",
    "67" : "g",
    "48" : "H",
    "68" : "h",
    "49" : "I",
    "69" : "i",
    "4a" : "J",
    "6a" : "j",
    "4b" : "K",
    "6b" : "k",
    "4c" : "L",
    "6c" : "l",
    "4d" : "M",
    "6d" : "m",
    "4e" : "N",
    "6e" : "n",
    "4f" : "O",
    "6f" : "o",
    "50" : "P",
    "70" : "p",
    "51" : "Q",
    "71" : "q",
    "52" : "R",
    "72" : "r",
    "53" : "S",
    "73" : "s",
    "54" : "T",
    "74" : "t",
    "55" : "U",
    "75" : "u",
    "56" : "V",
    "76" : "v",
    "57" : "W",
    "77" : "w",
    "58" : "X",
    "78" : "x",
    "59" : "Y",
    "79" : "y",
    "5a" : "Z",
    "7a" : "z",
    "30" : "0",
    "31" : "1",
    "32" : "2",
    "33" : "3",
    "34" : "4",
    "35" : "5",
    "36" : "6",
    "37" : "7",
    "38" : "8",
    "39" : "9",
}


def format_image(image:Image.Image) -> np.array:
    """
    Formats the image into the required 28x28 grayscale format required by the MNIST classifier and returns the formatted image.

    Args:
        image (Image.Image): The image to be formatted

    Returns:
        np.array: a 1x784 numpy array corresponding to the intensity values on a scale of 0 to 1.
    """
    image = image.resize((28,28)).convert('L')
    arr = np.array(image)
    arr = arr/255
    arr = arr.flatten()
    arr = arr.tolist()
    return arr


def download_data():
    """
    Creates a Downloaded_files subdirectory in files if not present then downloads the dataset
    """
    print("Creating Directory")
    Path("/opt/airflow/files/Downloaded_files").mkdir(parents=True, exist_ok=True)
    print("Created Downloaded_files Directory")
    urlretrieve(data_url, download_path)

def unzip_data():
    """
     Creates an extracted_files subdirectory in files if not present then extracts the {class}/train_{class} images to them
    """
    # os.makedirs("extracted_files/", exist_ok=True)
    Path("/opt/airflow/files/extracted_files").mkdir(parents=True, exist_ok=True)  
    with ZipFile(download_path, 'r') as zip:
        lst = zip.namelist()
        lst = [path for path in lst if re.search('by_class\/[A-Za-z0-9]{2}\/train\_[A-Za-z0-9]{2}\/.+\.png', path)]
        for path in lst:
            os.makedirs(f'/opt/airflow/files/extracted_files/by_class/{ASCII_dict[path[9:11]]}', exist_ok = True)
            name = path.split('/')[-1]
            with open(f'/opt/airflow/files/extracted_files/by_class/{ASCII_dict[path[9:11]]}/{name}', 'wb') as f:
                f.write(zip.read(path))        
    
        
def preprocess_data():
    """
     Creates a preprocessed_files subdirectory in files if not present then reads each image in extracted_files,
     converts all the images into 1-D vectors and saves them all in a CSV file.
    """
    # os.makedirs('preprocessed_files', exist_ok=True)
    Path("/opt/airflow/files/preprocessed_files").mkdir(parents=True, exist_ok=True)
    files_list = os.listdir('/opt/airflow/files/extracted_files/by_class')
    full_df = pd.DataFrame(columns=list(range(784)) + ['Label'])
    for file in files_list:
        values = []
        images = os.listdir(f'/opt/airflow/files/extracted_files/by_class/{file}')
        for image in images:
            img = Image.open(f'/opt/airflow/files/extracted_files/by_class/{file}/{image}')
            image_array = format_image(img)
            values.append(image_array)
        try:
            values = values[:2400]
        except:
            values = values
        df = pd.DataFrame(values)
        df['label'] = file
        full_df = pd.concat([full_df, df], axis=0)

    full_df.to_csv(f'/opt/airflow/files/preprocessed_files/final_output.csv')

def zip_output():
    """
    zips the output csv file and then deletes all the intermediate folders and files.
    """
    #Uncomment to instead save a zip file
    # zip = ZipFile("/opt/airflow/files/preprocessed_files/final_output.zip", "w", ZIP_DEFLATED)
    # zip.write("/opt/airflow/files/preprocessed_files/final_output.csv")
    # zip.close()
    try:
        # os.remove("/opt/airflow/files/preprocessed_files/final_output.csv")
        os.rmdir("/opt/airflow/files/extracted_files")
        os.rmdir("/opt/airflow/files/Downloaded_files")
    except:
        pass
    
with DAG('fetch_data_files', start_date=datetime(2016, 1, 1), default_args={"retries" : 0}, catchup = False) as dag:
    
    #creates tasks based on the number of workers specified. I have specified 6.
    get_files_task = PythonOperator(
            task_id = f'get_files',
            python_callable = download_data,

    )
    unzip_task = PythonOperator(
        task_id = 'unzip_files',
        python_callable = unzip_data
    )
    
    preprocess_task = PythonOperator(
        task_id = 'preprocess_data',
        python_callable = preprocess_data
    )
    zip_task = PythonOperator(
        task_id = 'zip_output',
        python_callable = zip_output
    )

get_files_task >> unzip_task >> preprocess_task >> zip_task
