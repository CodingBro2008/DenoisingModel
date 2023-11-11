import torch
from torch.optim import Adam
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os

# Custom dataset loader
class DenoisingDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, f))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_name)
        image = image.convert('RGB')  # Convert to RGB if not already
        
        # Split image into non-overlapping patches
        patches = [image.crop((i, j, i+256, j+256)) for i in range(0, image.size[0], 256) for j in range(0, image.size[1], 256)]
        
        if self.transform:
            patches = [self.transform(patch) for patch in patches]
        
        # Stack patches into a single tensor
        patches = torch.stack(patches)
        return patches


# Channel Attention module 
class ChannelAttention(nn.Module):
    def __init__(self, in_planes):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out) * x

# Multi-Scale Denoising Module
class MultiscaleModule(nn.Module):
    def __init__(self, in_channels):
        super(MultiscaleModule, self).__init__()
        # Branch 1 with regular convolution
        self.branch1_conv1 = ConvBlock(in_channels, in_channels)
        self.branch1_conv2 = ConvBlock(in_channels, in_channels)
        
        # Branch 2 with dilated convolution
        self.branch2_conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=2, dilation=2)
        self.branch2_conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=2, dilation=2)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        # Branch 1
        branch1_out = self.branch1_conv2(self.branch1_conv1(x))
        
        # Branch 2
        branch2_out = self.relu(self.branch2_conv2(self.relu(self.branch2_conv1(x))))
        
        # Element-wise addition of the two branches
        out = branch1_out + branch2_out
        return out



# Half-Instance Normalization (HIN) Layer
class HalfInstanceNormalization(nn.Module):
    def __init__(self, num_features):
        super(HalfInstanceNormalization, self).__init__()
        self.num_features = num_features
        # Instance Normalization applied to the first half of the channels
        self.instance_norm = nn.InstanceNorm2d(num_features // 2, affine=True)

    def forward(self, x):
        # Split the channels into two halves
        x1, x2 = x.chunk(2, dim=1)
        # Apply Instance Normalization to the first half
        x1 = self.instance_norm(x1)
        # Concatenate the normalized and non-normalized halves
        x = torch.cat((x1, x2), dim=1)
        return x

# Modified Residual Module with HIN and LeakyReLU
class ResidualModule(nn.Module):
    def __init__(self, in_channels):
        super(ResidualModule, self).__init__()
        # Convolutional layer followed by HIN and LeakyReLU
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.hin = HalfInstanceNormalization(in_channels)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        # Second convolutional layer without any normalization
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.hin(out)
        out = self.lrelu(out)
        out = self.conv2(out)
        return out + residual


# Define the denoising neural network model
class DenoisingNet(nn.Module):
    def __init__(self):
        super(DenoisingNet, self).__init__()
        # Noise level estimation stage
        self.noise_estimation = nn.Sequential(
            ConvBlock(3, 64),
            ConvBlock(64, 64),
            ConvBlock(64, 64),
            ConvBlock(64, 64),
            ChannelAttention(64),
            ConvBlock(64, 64)
        )
        # Non-blind denoising stage
        self.non_blind_denoising = nn.Sequential(
            ConvBlock(64, 64),
            MultiscaleModule(64),
            MultiscaleModule(64),
            MultiscaleModule(64),
            ResidualModule(64),
            ResidualModule(64),
            ConvBlock(64, 64),
            ConvBlock(64, 64)
        )

    def forward(self, x):
        x = self.noise_estimation(x)
        x = self.non_blind_denoising(x)
        return x

# Function for training the network
def train_model(model, dataloader, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        for patches in dataloader:
            # Assuming that 'patches' is a batch of noisy images and we have corresponding ground truth clean images
            # You will need to provide the ground truth images for the loss calculation
            # For example, if ground truth images are in a separate dataset
            # ground_truth_patches = ground_truth_dataset[idx]

            # Forward pass
            outputs = model(patches)
            loss = torch.mean(torch.abs(outputs - patches))  # L1 loss calculation

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

# Function for testing the network
def test_model(model, dataloader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for patches in dataloader:
            # Assuming that 'patches' is a batch of noisy images and we have corresponding ground truth clean images
            # You will need to provide the ground truth images for the evaluation
            # For example, if ground truth images are in a separate dataset
            # ground_truth_patches = ground_truth_dataset[idx]

            # Forward pass
            outputs = model(patches)
            loss = torch.mean(torch.abs(outputs - patches))
            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f'Average L1 Loss on the test set: {avg_loss:.4f}')
    return avg_loss

# Example usage
# Assuming we have separate dataloaders for training and testing
train_dataset = DenoisingDataset(root_dir='path_to_training_dataset', transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)

test_dataset = DenoisingDataset(root_dir='path_to_test_dataset', transform=transform)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

model = DenoisingNet()
optimizer = Adam(model.parameters(), lr=0.001)  # Example optimizer

num_epochs = 10  # Example number of epochs
train_model(model, train_dataloader, optimizer, num_epochs)
test_model(model, test_dataloader)