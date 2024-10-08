import torch
import torch.nn as nn


class Detection_network(nn.Module):
    def __init__(self, num_classes=3):
        super(Detection_network, self).__init__()

        self.bloc1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),  # Shape (53,53)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))  # Shape (27, 27)

        self.bloc2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))  # Shape (13, 13)

        self.bloc3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        self.bloc4 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))  # Shape (6, 6)

        self.fc = nn.Sequential(
            nn.Linear(32 * 6 * 6, 100),
            nn.ReLU())

        # Output size is 3 objects * (4 parameters + 3 class logits) = 3 * 7 = 21
        self.fc1 = nn.Linear(100, 3 * (4 + num_classes))  # Output shape: (batch_size, 21)

    def forward(self, x):
        x = self.bloc1(x)
        x = self.bloc2(x)
        x = self.bloc3(x)
        x = self.bloc4(x)
        x = x.reshape(x.size(0), -1)  # Flatten to (batch_size, 32 * 6 * 6)
        x = self.fc(x)
        x = self.fc1(x)  # Output shape: (batch_size, 21)

        # Reshape output to (batch_size, 3, 7) where 7 is (4 parameters + 3 logits)
        output = x.view(x.size(0), 3, 4 + 3)  # Shape: (batch_size, 3, 7)

        # Apply sigmoid activation to the first 4 parameters (presence, x, y, size)
        output[:, :, :4] = torch.sigmoid(output[:, :, :4])

        # The last 3 values (logits) are kept as they are for class prediction
        # (No activation applied here since they will be used with CrossEntropyLoss)

        return output



class AlexNet(nn.Module):
    def __init__(self, num_classes=3):
        super(AlexNet, self).__init__()

        # Since the input image size is 53x53, there is no need for upscaling at the start.

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))  # Output size: (16, 27, 27)

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))  # Output size: (16, 13, 13)

        self.layer4 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU())  # Output size: (32, 13, 13)

        self.layer5 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))  # Output size: (32, 6, 6)

        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(32 * 6 * 6, 100),
            nn.ReLU())

        # Output size is 3 objects * (4 parameters + 3 class logits) = 3 * 7 = 21
        self.fc1 = nn.Linear(100, 3 * (4 + num_classes))  # Output shape: (batch_size, 21)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = out.reshape(out.size(0), -1)  # Flatten to (batch_size, 32 * 6 * 6)
        out = self.fc(out)
        out = self.fc1(out)  # Output shape: (batch_size, 21)

        # Reshape output to (batch_size, 3, 7) where 7 is (4 parameters + 3 logits)
        out = out.view(out.size(0), 3, 4 + 3)  # Shape: (batch_size, 3, 7)

        # Apply sigmoid activation to the first 4 parameters (presence, x, y, size)
        out[:, :, :4] = torch.sigmoid(out[:, :, :4])

        # The last 3 values (logits) are kept as they are for class prediction
        # (No activation applied here since they will be used with CrossEntropyLoss)

        return out
#marche pas :
#
#
#
# class Detection_network(nn.Module):
#     def __init__(self, inputs_channels, n_classes):
#         super(Detection_network, self).__init__()
#
#         # Inspired AlexNet network
#         # Feature extraction
#         # Bloc1
#         self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=11, stride=2, padding=4) # Output 26x26
#         self.relu1 = nn.ReLU()
#         self.maxpol1 = nn.MaxPool2d(kernel_size=2, padding=0, stride=2)                             # Output 13x13
#         # Bloc3
#         self.conv3 = nn.Conv2d(in_channels=8, out_channels=32, kernel_size=3, stride=1, padding=1)
#         self.relu3 = nn.ReLU()
#         # Bloc5
#         self.conv5 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
#         self.relu5 = nn.ReLU()
#         self.maxpol5 = nn.MaxPool2d(kernel_size=2, padding=0, stride=2)                             # Output 6x6
#
#         # Classification
#         # Bloc 6
#         self.fn6 = nn.Flatten()
#         self.fl6 = nn.Linear(32*6*6, 256)
#         self.relu6 = nn.ReLU()
#
#         # Bloc 7
#         self.fl7 = nn.Linear(256, 256)
#         self.relu7 = nn.ReLU()
#
#         # Bloc 8
#         self.fl8 = nn.Linear(256, 5)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.relu1(x)
#         x = self.maxpol1(x)
#
#         x = self.conv3(x)
#         x = self.relu3(x)
#
#         x = self.conv5(x)
#         x = self.relu5(x)
#         x = self.maxpol5(x)
#
#         x = self.fn6(x)
#         x = self.fl6(x)
#         x = self.relu6(x)
#
#         x = self.fl7(x)
#         x = self.relu7(x)
#
#         output = self.fl8(x)
#
#         return output
