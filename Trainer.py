import torch as torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

class Trainer:
    def __init__(self,
                 model: torch.nn.Module,
                 loss_fn: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 calculate_accuracy,
                 device: torch.device,
                 loss_steps: int = 100):
        
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.calculate_accuracy = calculate_accuracy
        self.device = device
        self.loss_steps = loss_steps

    def train(self, data_loader: torch.utils.data.DataLoader, epoch=None):
        #Training
        train_loss, train_acc = 0, 0
        #Put Data into training Mode
        self.model.train()
        for batch, (X, y) in enumerate(data_loader):
            X, y = X.to(self.device), y.to(self.device)
            y_pred = self.model(X)
            loss = self.loss_fn(y_pred, y)
            train_loss += loss.item()
            train_acc += self.calculate_accuracy(y_true=y,
                                                 y_pred=y_pred.argmax(dim=1)) #from logits -> prediction labels
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        train_loss = train_loss / len(data_loader)
        train_acc = train_acc / len(data_loader)
        if epoch is not None and epoch % self.loss_steps == 0:
            print(f"Training Loss: {train_loss:.5f} | Training Accuracy: {train_acc:.5f}%")
        return train_loss, train_acc

    def test(self, data_loader: torch.utils.data.DataLoader, epoch=None):
        #Testing
        test_loss, test_acc = 0, 0
        self.model.to(self.device)
        #Put Data into evaluation Mode
        self.model.eval()
        with torch.inference_mode():
            for X, y in data_loader:
                X, y = X.to(self.device), y.to(self.device)
                test_pred = self.model(X)
                loss = self.loss_fn(test_pred, y)
                test_loss += loss.item()
                test_acc += self.calculate_accuracy(y_true=y, y_pred=test_pred.argmax(dim=1))

            test_loss = test_loss / len(data_loader)
            test_acc = test_acc / len(data_loader)
            if epoch is not None and epoch % self.loss_steps == 0:
                print(f"Test Loss {test_loss:.5f} | Test Accuracy {test_acc:.5f}%")
            return test_loss, test_acc