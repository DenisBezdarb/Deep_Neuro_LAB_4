import torch
import torch.nn as nn
import pandas as pd

n = 12
df = pd.read_csv('dataset_simple.csv')
if n % 2 == 1:
    # Классификация: предсказание will_buy по age и income
    X = torch.Tensor(df[['age', 'income']].values)
    y = torch.Tensor(df['will_buy'].values).view(-1, 1)

    class ClassificationNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(nn.Linear(2, 5),
                nn.Tanh(),
                nn.Linear(5, 1),
                nn.Sigmoid()
            )
        
        def forward(self, x):
            return self.layers(x)
    
    model = ClassificationNet()
    loss_fn = nn.BCELoss()
    task_type = 'classification'

else:
    # Регрессия: предсказание income по age
    X = torch.Tensor(df[['age']].values)
    y = torch.Tensor(df['income'].values).view(-1, 1)

    X = (X - X.mean()) / X.std()
    y = (y - y.mean()) / y.std()

    class RegressionNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(1, 10),
                nn.ReLU(),
                nn.Linear(10, 1)
            )
        
        def forward(self, x):
            return self.layers(x)
    
    model = RegressionNet()
    loss_fn = nn.MSELoss()
    task_type = 'regression'

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
epochs = 100

for epoch in range(epochs):
    pred = model(X)
    loss = loss_fn(pred, y)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item():.4f}')

if task_type == 'classification':
    with torch.no_grad():
        predictions = (model(X) > 0.5).float()
    accuracy = (predictions == y).float().mean()
    print(f'\nТочность классификации: {accuracy.item()*100:.2f}%')

else:
    with torch.no_grad():
        predictions = model(X)
    mae = torch.mean(torch.abs(predictions - y)).item()
    print(f'\nСредняя абсолютная ошибка (MAE): {mae:.2f}')