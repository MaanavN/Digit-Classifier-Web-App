import torch
import torch.nn as nn



class DigitNet(nn.Module):
    def __init__(self, input_size, hidden1_size, hidden2_size, output_size):
        super(DigitNet, self).__init__()
        
        
        self.relu = nn.ReLU()
        
        self.fc1 = nn.Linear(input_size, hidden1_size)
        self.fc2 = nn.Linear(hidden1_size, hidden2_size)
        self.fc3 = nn.Linear(hidden2_size, output_size)
        
    def forward(self, x):
        x = x.view(-1, 784)
        
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        
        return out
    

model = DigitNet(784, 32, 16, 10)
model.load_state_dict(torch.load('model.pth'))



def classify_image(image):
    model.eval()
    
    output = model.forward(image)
    highest_val = sorted(output.tolist()[0])[-1]
    output = (output.tolist()[0]).index(highest_val)

    
    return output