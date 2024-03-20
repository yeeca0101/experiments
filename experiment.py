import os
from time import time
import torch


class Trainer:
    def __init__(self, model, data_loader, test_loader, criterion, optimizer, device):
        self.model = model
        self.data_loader = data_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

        self.model.to(device)

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

        self.results = {
            'train_loss_history': train_loss_history,
            'test_loss_history': test_loss_history,
            'test_accuracy_history': test_accuracy_history,
            'train_test_time': f"{minutes}m{seconds}s"
        }
        return self.results

    def run_pipline(self,activation_functions,num_epochs,experiments_dir):
        os.makedirs(experiments_dir,exist_ok=True)

        # build model 

        # 
        for name, activation_function in activation_functions.items():
            print(f"Training with {name} activation function...")
            self.run(num_epochs=num_epochs)

        self.visualize_result(experiments_dir,save=True)

    def visualize_result(self,experiments_dir=None,save=False):
        import matplotlib.pyplot as plt

        save_re = True if save and os.path.isdir(experiments_dir) else False

        plt.figure()
        for name, data in self.results.items():
            plt.plot(data['train_loss_history'], label=f"{name}:{data['train_test_time']}")
        plt.xlabel('Epoch')
        plt.ylabel('Training Loss')
        plt.legend()
        if save_re : plt.savefig(os.path.join(experiments_dir,f'train_loss_{"_".join(self.results.keys())}.png'))
        plt.show()
        
        # Plot the testing loss
        plt.figure()
        for name, data in self.results.items():
            plt.plot(data['test_loss_history'], label=name)
        plt.xlabel('Epoch')
        plt.ylabel('Testing Loss')
        plt.legend()
        if save_re : plt.savefig(os.path.join(experiments_dir,f'test_loss_{"_".join(self.results.keys())}.png'))
        plt.show()
        
        # Plot the testing accuracy
        plt.figure()
        for name, data in self.results.items():
            plt.plot(data['test_accuracy_history'], label=name)
        plt.xlabel('Epoch')
        plt.ylabel('Testing Accuracy')
        plt.legend()
        if save_re : plt.savefig(os.path.join(experiments_dir,f'test_acc_{"_".join(self.results.keys())}.png'))
        plt.show()