import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import os
from dataset import *
import numpy as np
def main():
    def calculate_score(loader, model):
        model.eval()  # Set the model to evaluation mode
        label_path = os.path.join(os.getcwd(),'..','data','NHANES2','point_predictor_data_v2','labels.pkl')
        with open(label_path,'rb') as f:
            all_labels = pickle.load(f)
        lossfunc = nn.MSELoss()
        total_loss = 0
        
        with torch.no_grad():  # No need to compute gradients during validation
            for images, ids in loader:
                labels = [all_labels[id] for id in ids]
                images, labels = torch.Tensor(np.array(images)).to(device), torch.Tensor(np.array(labels)).to(device=device,dtype=torch.float32)
                outputs = model(images)
                loss = lossfunc(outputs,labels)
                total_loss += loss

        return total_loss
    root = os.path.join(os.getcwd(),'..')
    data_path = os.path.join(root,'data','NHANES2','point_predictor_data_v2')
    label_path = os.path.join(os.getcwd(),'..','data','NHANES2','point_predictor_data_v2','labels.pkl')
    with open(label_path,'rb') as f:
        all_labels = pickle.load(f)

    train_loader = get_data_loader(dataset='PointPredictorV2',data_path=data_path,mode='Training',b=32)
    val_loader = get_data_loader(dataset='PointPredictorV2',data_path=data_path,mode='Validation',b=16)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load a pre-trained ResNet model
    ResNet = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights)

    # Reconfigure the final layer for regression
    in_features = ResNet.fc.in_features
    ResNet.fc = torch.nn.Linear(in_features,2)

    ResNet.to(device)

    # Define the loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(ResNet.parameters(), lr=0.001)

    num_epochs = 15
    best_accuracy = 0
    big_list =[]
    
    for epoch in range(num_epochs):

        running_loss = 0
        ResNet.train()
        biggest_ids = []
        biggest_loss = 0

        for i, (images, ids) in enumerate(train_loader):
            # Move inputs and labels to the device
            labels = [all_labels[id] for id in ids]
            images, labels = torch.Tensor(np.array(images)).to(device=device,dtype=torch.float32), torch.Tensor(np.array(labels)).to(device=device,dtype=torch.float32)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = ResNet(images)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            if loss > biggest_loss:
                biggest_loss = loss
                biggest_ids = ids

            running_loss += loss.item()
            
                
            if i % 10 == 9:  # Print every 10 mini-batches
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {running_loss / 10:.4f}')
                running_loss = 0.0
        
        if big_list == []:
            big_list = biggest_ids
        else:
            big_list = [id for id in big_list if id in biggest_ids]
        
        print(big_list)
        val_accuracy = calculate_score(val_loader, ResNet)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Validation Accuracy: {val_accuracy:.4f}')
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(ResNet.state_dict(),os.path.join('weights','point_predictor_v2'))



if __name__ == '__main__':
    main()