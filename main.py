import streamlit as st
import os
import numpy as np
import pandas as pd
import torch.nn as nn
import torch
from PIL import Image
from model import UNet
from dataset import SyntheticCellDataset
from model import UNet
from loss import DiceLoss
import torch
from utils import save_checkpoint, load_checkpoint
from torchvision import transforms
import torchvision.transforms.functional as TF
from torchvision import transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#st.set_option('deprecation.showfileUploaderEncoding', False)
st.title("Unet Visualisation")
st.markdown('By DRIP AI TEAM')

add_selectbox = st.sidebar.selectbox(
    "What action do you want to perform",
    ("Training", "Evaluation")
)

def train(model, train_loader, device, optimizer):
    model.train()
    steps = len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)
    train_loss = 0.0
    dsc_loss = DiceLoss()

    progress_bar = st.sidebar.progress(0)

    for i, data in enumerate(train_loader):
        x,y = data

        optimizer.zero_grad()
        y_pred = model(x.to(device))
        loss = dsc_loss(y_pred, y.to(device))
        train_loss_list.append(loss.item())
        train_loss_detail.line_chart(np.array(train_loss_list))
        progress_bar.progress((i+1)/len(train_loader))

        loss.backward()
        optimizer.step()
        scheduler.step()

        train_loss += loss.item()
    return model, train_loss/len(train_loader), optimizer

def validate(model, val_loader, device):

    with torch.no_grad():
        model.eval()
        val_loss = 0.0

        for i, data in enumerate(val_loader):
            x,y = data

            y_pred = model(x.to(device))
            loss = dsc_loss(y_pred, y.to(device))


            val_loss += loss.item()
    return val_loss/len(val_loader)

def display_dataset_details(dataset, train_set, val_set):
    x = 'Total Images :'+str(len(dataset))
    y = 'Total Training Images :'+str(len(train_set))
    z = 'Total Validation Images :'+str(len(val_set))

    return x,y,z

if(add_selectbox == 'Training'):
    st.header(add_selectbox)
    st.markdown('Device Detected : '+str(device))
    st.write('Select Training Parameters')
    epochs = st.number_input('Epochs', min_value = 1, value = 2)
    lr = st.number_input('Learning Rate', min_value = 0.0001, max_value = None, value = 0.0010, step = 0.001, format = '%f')

    if st.button('Load Data'):
        dsc_loss = DiceLoss()
        dataset = SyntheticCellDataset('dataset/image', 'dataset/mask')
        indices = torch.randperm(len(dataset)).tolist()
        sr = int(0.2 * len(dataset))
        train_set = torch.utils.data.Subset(dataset, indices[:-sr])
        val_set = torch.utils.data.Subset(dataset, indices[-sr:])
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=2, shuffle=True, pin_memory=True)
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=2, shuffle=False, pin_memory=True)
        st.write('Data Loaded')
        x,y,z = display_dataset_details(dataset, train_set, val_set)
        st.write(x)
        st.write(y)
        st.write(z)
        #if st.button('Start Training'):
        model = UNet()
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr)
        st.markdown('Training Started')
        overall_train = []
        overall_val = []
        # st.bar_chart([overall_train, overall_val])
        chart_data = pd.DataFrame(np.array([overall_train, overall_val]).transpose(), columns=['Train', 'Val'])
        #chart_data.columns = ['Train','Val']
        #st.line_chart(chart_data)
        train_chart = st.empty()
        train_bar = st.empty()
        val_overall = 1000
        train_loss_list = []
        train_loss_detail = st.sidebar.empty()
        for epoch in range(epochs):
            model, train_loss, optimizer = train(model, train_loader, device, optimizer)
            val_loss = validate(model, val_loader, device)
            overall_train.append(train_loss)
            overall_val.append(val_loss)
            chart_data = pd.DataFrame(np.array([overall_train, overall_val]).transpose(), columns=['Train', 'Val'])
            train_chart.line_chart(chart_data)
            train_bar.bar_chart(chart_data)
            if val_loss < val_overall:
                save_checkpoint('./weight//epoch_'+str(epoch+1), model, train_loss, val_loss, epoch)
                val_overall = val_loss

            st.write('[{}/{}] train loss :{} val loss : {}'.format(epoch+1, epochs, train_loss, val_loss))

        else:
            print('kichi gote issue achi')

if(add_selectbox == 'Evaluation'):
    st.header(add_selectbox)
    st.markdown('Device Detected : '+str(device))
    weights = sorted(os.listdir('./weight'), key=len)
    selected_weight = st.selectbox('Select a trained model', weights)

    if selected_weight is not None:
        st.markdown('Selected Trained Model :'+ str(selected_weight))
        model = UNet()
        model.to(device)
        load_checkpoint('./weight/'+str(selected_weight), model, device=device)
        model.eval()
        uploaded_file = st.file_uploader("Upload an Image", type=['png', 'PNG'])
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert('L')
            resize = transforms.Resize(size=(256, 256))
            image = resize(image)
            st.write('Input Image')
            st.image(image, caption='Uploaded Image.', use_column_width=True)
            tensor_img = TF.to_tensor(image)
            with torch.no_grad():
                output = model(tensor_img.unsqueeze(0).to(device))
            out_img = output.squeeze(0).cpu().detach().numpy().transpose(1,2,0)
            st.image(out_img, caption='Predicted', use_column_width=True)
