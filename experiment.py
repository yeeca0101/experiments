import torch
from time import time


class Trainer:
    def __init__(self, model, data_loader, test_loader, criterion, optimizer, device):
        self.model = model
        self.data_loader = data_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

    def train(self):
        self.model.train()
        running_loss = 0.0

        for data, target in self.data_loader:
            data, target = data.to(self.device), target.to(self.device)
            data = data.view(data.shape[0], -1)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * data.size(0)

        return running_loss / len(self.data_loader.dataset)

    def test(self):
        self.model.eval()
        correct = 0
        total = 0
        test_loss = 0.0

        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                data = data.view(data.shape[0], -1)

                output = self.model(data)
                loss = self.criterion(output, target)
                test_loss += loss.item() * data.size(0)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        return test_loss / len(self.test_loader.dataset), 100 * correct / total

    def run(self, num_epochs):
        train_loss_history = []
        test_loss_history = []
        test_accuracy_history = []

        start_t = time()
        for epoch in range(num_epochs):
            train_loss = self.train()
            test_loss, test_accuracy = self.test()

            train_loss_history.append(train_loss)
            test_loss_history.append(test_loss)
            test_accuracy_history.append(test_accuracy)

            print(f"Epoch [{epoch+1}/{num_epochs}], Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")
        
        end_t = time() - start_t
        minutes = int(end_t // 60)
        seconds = int(end_t % 60)

        return {
            'train_loss_history': train_loss_history,
            'test_loss_history': test_loss_history,
            'test_accuracy_history': test_accuracy_history,
            'train_test_time': f"{minutes}m{seconds}s"
        }
