import torch
import os
import streamlit as st

def save_checkpoint(save_path, model, train_loss, val_loss, start_epoch):

    if save_path == None:
        return

    state_dict = {'model_state_dict': model.state_dict(),
                  'train_loss': train_loss,
                  'val_loss' : val_loss,
                  'start_epoch' : start_epoch}
    #print(os.path.join(os.getcwd(),save_path))
    torch.save(state_dict, save_path)
    print(f'Model saved to ==> {save_path}')

def load_checkpoint(load_path, model, device='cpu'):

    if load_path==None:
        return

    state_dict = torch.load(load_path, map_location=device)
    print(f'Model loaded from <== {load_path}')

    model.load_state_dict(state_dict['model_state_dict'])
    st.write('epoch is ', state_dict['start_epoch']+1)
    st.write('train_loss is ', state_dict['train_loss'])
    st.write('val_loss is ', state_dict['val_loss'])
