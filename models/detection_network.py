import torch
import torch.nn as nn
import torchvision.models as models



class YOLO(nn.Module):
    def __init__(self, num_classes=3):
        super(YOLO, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=12, kernel_size=3, stride=1, padding=1),
            # Reduced filters from 16 to 12
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2d(kernel_size=2, padding=0, stride=2),  # Dim = 26x26

            # Bloc2
            nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, stride=1, padding=1),
            # Reduced filters from 32 to 24
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=24, out_channels=24, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=24, out_channels=24, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2d(kernel_size=2, padding=0, stride=2),  # Dim = 13x13

            # Bloc3
            nn.Conv2d(in_channels=24, out_channels=48, kernel_size=3, stride=1, padding=1),
            # Reduced filters from 64 to 48
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=48, out_channels=48, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=48, out_channels=48, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1),

            # Flatten and fully connected layers
            nn.Flatten(),
            nn.Linear(48 * 13 * 13, 24),  # Reduced output units from 32 to 24
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(24, 7 * 7 * 15)
        )

        # Sigmoid activation to squash output between 0 and 1 for bounding box coordinates and presence
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        # x = self.feature_extractor(x)
        # x = self.conv_layer(x)
        # x = self.batch_norm_conv(x)
        # x = self.relu(x)
        # x = x.view(x.size(0), -1)  # Reshape before fully connected
        # x = self.fc_layers(x)
        # x = x.view(x.size(0), 3, 5)  # Reshape to (batch_size, 3, 5)
        #
        # # Apply activations
        # x[:, :, 0] = torch.sigmoid(x[:, :, 0])  # Presence (binary classification)
        # x[:, :, 1:3] = torch.sigmoid(x[:, :, 1:3])  # Bounding box coordinates (normalized)
        # x[:, :, 4] = torch.argmax(torch.softmax(x[:, :, 4:], dim=-1), dim=-1).float()  # Class IDs (argmax over softmax probabilities)

        x = self.model(x)
        output = x.view(-1, 7, 7, 15)
        return output


