import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

df = pd.read_csv('Data/iris.csv')
features = df.drop('target', axis=1).values
targets = df['target'].values
x_train, x_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=23)
x_train = torch.FloatTensor(x_train)
x_test = torch.FloatTensor(x_test)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)


class Model(nn.Module):
    def __init__(self, in_f = 4, h1 = 10, h2 = 12, out_f = 3):
        super().__init__()
        self.fc1 = nn.Linear(in_f, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, out_f)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x


trainloader = DataLoader(x_train, batch_size=40, shuffle=True)
testloader = DataLoader(y_train, batch_size=40, shuffle=True)

model = Model()
criterion = nn.CrossEntropyLoss()
Optimizer = torch.optim.Adam(model.parameters(), lr=0.0126)

epoch = 150
losses = []
for i in range(epoch):
    i += 1
    y_pred = model.forward(x_train)
    loss = criterion(y_pred, y_train)
    losses.append(loss)
    print(f'Epoch {i}, the loss is {loss}')

    Optimizer.zero_grad()
    loss.backward()
    Optimizer.step()

with torch.no_grad():
    y_pred = model(x_test)
    loss = criterion(y_pred, y_test)
print(f'{loss:.8f}')

plt.plot(range(150), losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

count = 0
with torch.no_grad():
    for data_line, real_label in zip(x_test, y_test):
        if model(data_line).argmax().item() == real_label.numpy().tolist():
            count += 1
accuracy = count / len(x_test) * 100
print(f'The accuracy in the test set is {accuracy}%')

torch.save(model.state_dict(), 'a.pt')
