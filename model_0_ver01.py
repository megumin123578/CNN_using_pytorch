import torch
from torch import nn
import pandas as pd
import numpy as np
from torchvision import transforms
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List
from timeit import default_timer as timer
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision

import random
from PIL import Image
import glob
from pathlib import Path
import seaborn as sns



print(torch.__version__)
device =  'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

# ktra drive
import os
def check_file_path(dir_path):
  for dirpath, dirnames, filenames in os.walk(dir_path):
    print(f'gom co {len(dirnames)} duong dan va {len(filenames)} thu muc trong {dirpath}')
data_path = 'E:\IT\AI\Thực hành trí tuệ nhân tạo\B3\data'
check_file_path(data_path)

# thiet lap duong dan den train va test
train_dir = 'data/train'
test_dir = 'data/test'
image_path = 'data'
print(len(train_dir), len(test_dir))

# set seed
random.seed(102)
# 1.lấy toàn bộ file có đuôi jpg trong tệp
image_path_list= glob.glob(f"{image_path}/*/*/*.jpg")
random_image_path = random.choice(image_path_list)
image_class = Path(random_image_path).parent.stem
img = Image.open(random_image_path)
print(f"Random image path: {random_image_path}")
print(f"Image class: {image_class}")
print(f"Image height: {img.height}")
print(f"Image width: {img.width}")
# Hiển thị ảnh (mở trong trình xem ảnh của hệ điều hành)
# img.show()

# dua ve dang numpy
sns.set_theme()
img_as_array = np.asarray(img)

# Plot ảnh với matplotlib
plt.figure(figsize=(8, 6))
plt.imshow(img_as_array)
plt.title(f"Image class: {image_class} | Image shape: {img_as_array.shape} -> [height, width, color_channels]")
plt.axis(False)
# plt.show()


#data augmenatation data
data_transform = transforms.Compose([
    transforms.Resize((64,64), antialias= None),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor()
])

def plot_transformed_images(image_paths, transform, n=3, seed=101):
    random.seed(seed)
    random_image_paths = random.sample(image_paths, k=n)
    for image_path in random_image_paths:
        with Image.open(image_path) as f:
            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(f)
            ax[0].set_title(f"Original \nSize: {f.size}")
            ax[0].axis("off")
            
            # (PyTorch default is [C, H, W] but Matplotlib is [H, W, C])
            transformed_image = transform(f).permute(1, 2, 0)
            ax[1].imshow(transformed_image)
            ax[1].set_title(f"Transformed \nSize: {transformed_image.shape}")
            ax[1].axis("off")
            fig.suptitle(f"Class: {Path(image_path).parent.stem}", fontsize=16)
    plt.show()  

plot_transformed_images(image_path_list, transform=data_transform, n=3, seed=101)

# setup traintestset
train_data = datasets.ImageFolder(root=train_dir,
                                  transform = data_transform)
test_data = datasets.ImageFolder(root = test_dir,
                                 transform = data_transform)
print(f'train data:\n {train_data}\n test data:\n{test_data}')
# list name
class_names = train_data.classes
print(class_names)
print(len(train_data), len(test_data))

img, label = train_data[0][0], train_data[0][1]
print(f"Image tensor:\n{img}")
print(f"Image shape: {img.shape}")
print(f"Image datatype: {img.dtype}")
print(f"Image label: {label}")
print(f"Label datatype: {type(label)}")

# thay doi thu tu chieu cua anh
img_permute = img.permute(1, 2, 0)

print(f"anh goc shape: {img.shape} -> [kenh mau, cao, rong]")
print(f"sau khi doi: {img_permute.shape}")




class Model_0(nn.Module):
  def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
    super().__init__()
    self.conv_block_1 = nn.Sequential(
        nn.Conv2d(in_channels=input_shape,
                  out_channels=hidden_units,
                  kernel_size=3,
                  stride=1,
                  padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=hidden_units,
                  out_channels=hidden_units,
                  kernel_size = 2,
                  stride = 1,
                  padding = 1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2,
                      stride=2)


    )
    self.conv_block_2 = nn.Sequential(
        nn.Conv2d(hidden_units, hidden_units*2, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(hidden_units*2, hidden_units*2, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )

    self.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features = hidden_units*2*16*16,
                  out_features = output_shape)
    )
  def forward(self, x: torch.Tensor):
      x = self.conv_block_1(x)
      x = self.conv_block_2(x)
      x = self.classifier(x)
      return x

model_0 = Model_0(input_shape=3, # so kenh mau(RGB -> 3)
                  hidden_units=10,
                  output_shape=len(train_data.classes)).to(device)
print(model_0)
# check infor model

from torchinfo import summary
summary(model_0, input_size=[1, 3, 64, 64]) # test dữ liệu đầu vào

def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer):
  model.train()

  train_loss, train_acc = 0, 0
  for batch, (X,y) in enumerate(dataloader):
    X, y = X.to(device), y.to(device)

    # forward_pass
    y_pred = model(X)

    # cal loss
    loss = loss_fn(y_pred, y)
    train_loss += loss.item()
    # optimize zero grad
    optimizer.zero_grad()
    # loss backward
    loss.backward()

    # optimizer step
    optimizer.step()

    y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
    train_acc += (y_pred_class == y).sum().item()/len(y_pred)

  train_loss = train_loss / len(dataloader)
  train_acc = train_acc / len(dataloader)

  return train_loss, train_acc


def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module):
  model.eval()
  test_loss, test_acc = 0, 0
  with torch.inference_mode():
    for batch, (X, y) in enumerate(dataloader):

        X, y = X.to(device), y.to(device)


        test_pred_logits = model(X)


        loss = loss_fn(test_pred_logits, y)
        test_loss += loss.item()


        test_pred_labels = test_pred_logits.argmax(dim=1)
        test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))
  test_loss = test_loss / len(dataloader)
  test_acc = test_acc / len(dataloader)
  return test_loss, test_acc

def cross_entropy(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """
    Tính toán Cross Entropy Loss giữa nhãn thực tế và dự đoán.

    Args:
        y_pred (torch.Tensor): Xác suất dự đoán từ mô hình (đã qua softmax).
                              Dạng [batch_size, num_classes].
        y_true (torch.Tensor): Nhãn thực tế (dạng chỉ mục cho từng lớp).
                              Dạng [batch_size].

    Returns:
        torch.Tensor: Giá trị cross entropy loss trung bình trên batch.
    """

    batch_size = y_pred.shape[0]


    log_probs = torch.log(y_pred)

    loss = -log_probs[range(batch_size), y_true]

    #tinh trung binh
    return loss.mean()

from tqdm.auto import tqdm
def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module = nn.CrossEntropyLoss(),
          epochs: int = 5):
  results = {"train_loss": [],
      "train_acc": [],
      "test_loss": [],
      "test_acc": []
  }

  for epoch in tqdm(range(epochs)):
      train_loss, train_acc = train_step(model=model,
                                          dataloader=train_dataloader,
                                          loss_fn=loss_fn,
                                          optimizer=optimizer)
      test_loss, test_acc = test_step(model=model,
          dataloader=test_dataloader,
          loss_fn=loss_fn)


      print(
          f"Epoch: {epoch+1} | "
          f"train_loss: {train_loss:.4f} | "
          f"train_acc: {train_acc:.4f} | "
          f"test_loss: {test_loss:.4f} | "
          f"test_acc: {test_acc:.4f}"
      )
      results["train_loss"].append(train_loss)
      results["train_acc"].append(train_acc)
      results["test_loss"].append(test_loss)
      results["test_acc"].append(test_acc)


  return results
# Set random seeds
torch.manual_seed(42)
torch.cuda.manual_seed(42)

NUM_EPOCHS = 300
model_0 = Model_0(input_shape=3,
                  hidden_units=10,
                  output_shape=len(train_data.classes)).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model_0.parameters(), lr=0.001)

def plot_loss_curves(results: Dict[str, List[float]]):


    loss = results['train_loss']
    test_loss = results['test_loss']


    accuracy = results['train_acc']
    test_accuracy = results['test_acc']


    epochs = range(len(results['train_loss']))

    # Setup a plot
    plt.figure(figsize=(15, 7))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label='train_loss')
    plt.plot(epochs, test_loss, label='test_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()
    plt.show()
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label='train_accuracy')
    plt.plot(epochs, test_accuracy, label='test_accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend()
    plt.show()

def pred(image_path, model):

  custom_image = torchvision.io.read_image(str(image_path)).type(torch.float32)
  custom_image = custom_image / 255.

  #resize
  custom_image_transform = transforms.Compose([
    transforms.Resize((64, 64)),
  ])
  custom_image_transformed = custom_image_transform(custom_image)
  model.eval()
  with torch.inference_mode():
      # thêm chiều
    #   custom_image_transformed_with_batch_size = custom_image_transformed.unsqueeze(dim=0)

      # dự đoán
      custom_image_pred = model_0(custom_image_transformed.unsqueeze(dim=0).to(device))
  custom_image_pred_probs = torch.softmax(custom_image_pred, dim=1)
  custom_image_pred_label = torch.argmax(custom_image_pred_probs, dim=1)
  custom_image_pred_class = class_names[custom_image_pred_label.cpu()]
  return custom_image_pred_class
def main():
# chuyen du lieu vao dataloader

    train_dataloader = DataLoader(dataset=train_data,
                                batch_size=32,
                                num_workers=0,
                                shuffle=True)
    test_dataloader = DataLoader(dataset=test_data,
                                batch_size=32,
                                num_workers=0,
                                shuffle=False)

    start_time = timer()
    model_0_results = train(model=model_0,
                            train_dataloader=train_dataloader,
                            test_dataloader=test_dataloader,
                            optimizer=optimizer,
                            loss_fn=loss_fn,
                            epochs=NUM_EPOCHS)
    end_time = timer()
    print(f"Total training time: {end_time-start_time:.3f} seconds")
    #show kq
    plot_loss_curves(model_0_results)
    torch.save(model_0.state_dict(),'model_1.pth')

    print(pred('test_img/beef-steak.png', model_0), pred('test_img/anh_mau.png', model_0), pred('test_img/sushi.jpg', model_0))
if __name__ == '__main__':      
   main()