import Architecture as arch
import torch
from torchvision.transforms import Compose, ToTensor,Resize
import os

#put the test image folder path
Test_folder = ""

current_path = os.path.dirname(os.path.abspath(__file__))
data_root = os.path.join(current_path,"Dataset")
classes = os.listdir(data_root)
num_class = len(classes)
result_root = os.path.join(current_path,"Result")

# chage pt file name
pt_file = "example.pt"

model = arch.Resnet50(3, 10)
model.load_state_dict(torch.load(os.path.join(result_root,pt_file)))
category = {}
for i in range(num_class):
    category.update({i:classes[i]})


# Load and preprocess the input image

preprocess = Compose([
      # Resize to model input size
    ToTensor(),
    Resize((224,229))# Convert to tensor
         # Normalize the image
])

def test_image(image):
    input_tensor = preprocess(image)
    print(type(input_tensor), input_tensor.shape)
    input_batch = input_tensor.unsqueeze(0)  # Add a batch dimension
    # Set the model to evaluation mode
    model.eval()

    # Forward pass through the model
    with torch.no_grad():
        output = model(input_batch)
    return output
# Convert the output to probabilities or class predictions
# (depending on your model and task)

for i in (os.listdir(Test_folder)):
    image = f'{Test_folder}/{i}'
    output = test_image(image)
    predictions = torch.argmax(output, dim=1)
    print(output)
    print(category[predictions.item()])
