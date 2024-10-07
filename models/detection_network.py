import torch
import torch.nn as nn
import torchvision.models as models



class YOLO(nn.Module):
    def __init__(self, num_classes=3):
        super(YOLO, self).__init__()

        # REAL YOLO :
        # nn.Conv2d(1, 64, kernel_size=7, padding=3, stride=2),
        # nn.LeakyReLU(0.1),
        # nn.MaxPool2d(kernel_size=2, stride=2),
        #
        # nn.Conv2d(64, 192, kernel_size=3, padding=1, stride=1),
        #
        # nn.MaxPool2d(kernel_size=2,padding=0 ,stride=2),
        #
        # nn.Conv2d(192, 128, kernel_size=1, padding=0, stride=1),
        # nn.LeakyReLU(0.1),
        #
        # nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=1),
        # nn.LeakyReLU(0.1),
        #
        # nn.Conv2d(256, 256, kernel_size=1, padding=0, stride=1),
        # nn.LeakyReLU(0.1),
        #
        # nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=1),
        # nn.LeakyReLU(0.1),
        #
        # nn.MaxPool2d(kernel_size=2, stride=2),
        #
        # nn.Conv2d(512, 256, kernel_size=1, padding=0, stride=1),
        # nn.LeakyReLU(0.1),
        #
        # nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=1),
        # nn.LeakyReLU(0.1),
        #
        # nn.Conv2d(512, 256, kernel_size=1, padding=0, stride=1),
        # nn.LeakyReLU(0.1),
        #
        # nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=1),
        # nn.LeakyReLU(0.1),
        #
        # nn.Conv2d(512, 1024, kernel_size=3, padding=1, stride=1),
        # nn.LeakyReLU(0.1),
        #
        # nn.Conv2d(1024, 512, kernel_size=1, padding=0, stride=1),
        # nn.ReLU(),
        #
        # nn.Conv2d(4096, num_classes * 7, kernel_size=1, padding=0, stride=1),
        # #reshape here
        # nn.Sigmoid()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, stride=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=1),
            nn.LeakyReLU(0.1),
        )

        # Fully Connected Layers for Bounding Box and Class Prediction
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 6 * 6, 1024),
            nn.LeakyReLU(0.1),
            nn.Linear(1024, num_classes * 5 * 3),
            # 5 outputs per box: (x, y, w, h, confidence) for 3 boxes per grid cell
            nn.Sigmoid()
        )

    def forward(self, x):

        x = self.feature_extractor(x)
        x = self.fc_layers(x)
        x = x.view(x.size(0), -1, 5)  # Will be reshaped to (batch_size, num_boxes, 5)
        return x

# model = YOLO()
#
# #Calculate the number of parameters
# total_params = sum(p.numel() for p in model.parameters())
# print(f"Total number of parameters: {total_params}")

