import os
from time import time
import torch

from model.models import replace_activations

class ModelCard:
    def __init__(self,model_class,**kwargs) -> None:
        self.model_class = model_class
        self.kwargs = kwargs
        self.model = None # model_class(**kwargs)
    
    def update_kwargs(self,**change_kwargs):
        self.kwargs.update(change_kwargs)
        self.model = self.model_class(**self.kwargs)
        

class Trainer:
    def __init__(self, model_card:ModelCard, 
                 data_loader, 
                 test_loader, 
                 criterion, 
                 optimizer, 
                 optimizer_kwargs,
                 device):
        self.modelcard = model_card
        self.model = model_card.model
        self.data_loader = data_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer_kwargs = optimizer_kwargs
        self.optim_class = optimizer
        self.optimizer = None # self._set_optimizer(optim_class=optimizer,**self.optimizer_kwargs)
        self.device = device


        self.results = {}

    def _set_optimizer(self,**kwargs):
        self.optimizer = self.optim_class(self.model.parameters(),**kwargs)

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

        results = {
            'train_loss_history': train_loss_history,
            'test_loss_history': test_loss_history,
            'test_accuracy_history': test_accuracy_history,
            'train_test_time': f"{minutes}m{seconds}s"
        }
        return results

    def run_pipline(self,activation_functions,num_epochs,experiments_dir):
        # fixed model, change activations
        os.makedirs(experiments_dir,exist_ok=True)

        for name, activation_function in activation_functions.items():
            # model update
            self._model_act_change(activation=activation_function)
            self._set_optimizer(**self.optimizer_kwargs)
            # start
            print(f"Training with {name} activation function...")
            self.model.to(self.device)
            res = self.run(num_epochs=num_epochs)
            self.results[name] = res

        self.visualize_result(experiments_dir,save=True)

    def _model_act_change(self,activation):
        self.modelcard.update_kwargs(activation=activation)
        self.model = self.modelcard.model

    def visualize_result(self,experiments_dir=None,save=False):
        import matplotlib.pyplot as plt

        save_re = True if save and os.path.isdir(experiments_dir) else False
        print(save_re)
        
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