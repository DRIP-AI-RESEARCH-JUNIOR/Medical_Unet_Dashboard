# Medical_Unet_Dashboard

## All the dependencies needs to be installed

  ```pip install requirements.txt```  
## Create an empty folder 'weight'  
  ```mkdir weight```  
  
## Add the dataset folder  
### Name of the dataset folder should be 'dataset' and the Image folder should be 'image' and Mask folder should be 'mask'
### Dataset Folder should look something as shown  
![](https://github.com/DRIP-AI-RESEARCH-JUNIOR/Medical_Unet_Dashboard/blob/main/img/img_8.png)  
*Dataset Folder*  

### Overall Working folder structure must be as shown below  
![](https://github.com/DRIP-AI-RESEARCH-JUNIOR/Medical_Unet_Dashboard/blob/main/img/img_7.png)  
*Working Directory*  

### Then run the app  
    ```streamlit run main.py```  
    
### Open the network URL generted on any browser  
![](https://github.com/DRIP-AI-RESEARCH-JUNIOR/Medical_Unet_Dashboard/blob/main/img/img_1.png)  
*Landing Dashboard*  

### Set the training parameters by selecting number of epochs and learning rate  
![](https://github.com/DRIP-AI-RESEARCH-JUNIOR/Medical_Unet_Dashboard/blob/main/img/img_2.png)  

### The training must start automatically after loading data and the logs would be displayed  
![](https://github.com/DRIP-AI-RESEARCH-JUNIOR/Medical_Unet_Dashboard/blob/main/img/Screenshot%20(25).png)  

### Training and Validation loss per epoch  
![](https://github.com/DRIP-AI-RESEARCH-JUNIOR/Medical_Unet_Dashboard/blob/main/img/img_3.png)  

## Evaluation  

### Select the train checkpoint you want to visualise the results of and upload image to see it's mask  
![](https://github.com/DRIP-AI-RESEARCH-JUNIOR/Medical_Unet_Dashboard/blob/main/img/img_4.png)  

![](https://github.com/DRIP-AI-RESEARCH-JUNIOR/Medical_Unet_Dashboard/blob/main/img/img_6.png)  
