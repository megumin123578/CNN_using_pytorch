import torch
from torchvision import transforms, datasets
from tqdm.auto import tqdm
import torchvision
from torch import nn
import warnings
warnings.filterwarnings("ignore")

device =  'cuda' if torch.cuda.is_available() else 'cpu'
model_ver = int(input('chọn mô hình thứ: '))
# setup traintestset
train_dir = 'data/train'
test_dir = 'data/test'
#data augmenatation data
data_transform = transforms.Compose([
    transforms.Resize((64,64), antialias= None),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor()
])


train_data = datasets.ImageFolder(root=train_dir,
                                  transform = data_transform)
test_data = datasets.ImageFolder(root = test_dir,
                                 transform = data_transform)
# print(f'train data:\n {train_data}\n test data:\n{test_data}')
# list name
class_names = train_data.classes


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
# load model
model_name = f'model_{model_ver}.pth'
model_0.load_state_dict(torch.load(model_name, map_location=device))

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
      custom_image_pred = model(custom_image_transformed.unsqueeze(dim=0).to(device))
  custom_image_pred_probs = torch.softmax(custom_image_pred, dim=1)
  custom_image_pred_label = torch.argmax(custom_image_pred_probs, dim=1)
  custom_image_pred_class = class_names[custom_image_pred_label.cpu()]
  return custom_image_pred_class


print(pred('test_img/pizza_1.jpg', model_0))