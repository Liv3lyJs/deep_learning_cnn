import torch 
import torch.nn as nn


class UNet(nn.Module):
    def __init__(self, input_channels, n_classes):
        super(UNet, self).__init__()

        # Define the layers that will be used in the created U-Net
        # Down1
        self.convdown1_1 = nn.Conv2d(in_channels=input_channels, out_channels=16, kernel_size=3, padding=1, stride=1)
        self.reludown1_1 = nn.ReLU()
        self.convdown1_2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1, stride=1)
        self.reludown1_2 = nn.ReLU()
        # Down2
        self.maxpolldown2 = nn.MaxPool2d(kernel_size=3, padding=0, stride=2)
        self.convdown2_1 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1, stride=1)
        self.reludown2_1 = nn.ReLU()
        self.convdown2_2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1, stride=1)
        self.reludown2_2 = nn.ReLU()
        # Down3
        self.maxpolldown3 = nn.MaxPool2d(kernel_size=3, padding=0, stride=2)
        self.convdown3_1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=1)
        self.reludown3_1 = nn.ReLU()
        self.convdown3_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1)
        self.reludown3_2 = nn.ReLU()

        # Bottleneck
        self.maxpollbk = nn.MaxPool2d(kernel_size=2, padding=0, stride=2)
        self.convbk3_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=1)
        self.relubk3_1 = nn.ReLU()
        self.convbk3_2 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1, stride=1)
        self.relubk3_2 = nn.ReLU()

        # Up3
        self.maxpollup3 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=2, padding=0, stride=2)
        self.convup3_1 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1, stride=1)
        self.reluup3_1 = nn.ReLU()
        self.convup3_2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1, stride=1)
        self.reluup3_2 = nn.ReLU()
        # Up2
        self.maxpollup2 = nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=4, padding=0, stride=2)
        self.convup2_1 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1, stride=1)
        self.reluup2_1 = nn.ReLU()
        self.convup2_2 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1, stride=1)
        self.reluup2_2 = nn.ReLU()
        # Up1 
        self.maxpollup1 = nn.ConvTranspose2d(in_channels=16, out_channels=16, kernel_size=3, padding=0, stride=2)
        self.convup1_1 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1, stride=1)
        self.reluup1_1 = nn.ReLU()
        self.convup1_2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1, stride=1)
        self.reluup1_2 = nn.ReLU()

        # Output
        self.output_conv = nn.Conv2d(in_channels=16, out_channels=n_classes, kernel_size=1, padding=0, stride=1)

    def forward(self, x):
        # Encoder ________________________________________________________________________________
        # Down 1
        # Down 1
        down1 = self.convdown1_1(x)
        down1 = self.reludown1_1(down1)
        down1 = self.convdown1_2(down1)
        down1 = self.reludown1_2(down1) # Use this as skip connection for Up 1
        # Down 2
        down2 = self.maxpolldown2(down1)

        down2 = self.convdown2_1(down2)
        down2 = self.reludown2_1(down2)
        down2 = self.convdown2_2(down2)
        down2 = self.reludown2_2(down2) # Use this as skip connection for Up 2
        # Down 3
        down3 = self.maxpolldown3(down2)
        
        down3 = self.convdown3_1(down3)
        down3 = self.reludown3_1(down3)
        down3 = self.convdown3_2(down3)
        down3 = self.reludown3_2(down3) # Use this as skip connection for Up 3


        # Bottleneck _______________________________________________________________________________
        bottleneck = self.maxpollbk(down3)

        bottleneck = self.convbk3_1(bottleneck)
        bottleneck = self.relubk3_1(bottleneck)
        bottleneck = self.convbk3_2(bottleneck)
        bottleneck = self.relubk3_2(bottleneck)



        # Decoder _________________________________________________________________________________
        # Up 3
        up3 = self.maxpollup3(bottleneck)

        up3 = torch.cat([up3, down3], dim=1) # Skip connection concatenation along the channel axis
        up3 = self.convup3_1(up3)
        up3 = self.reluup3_1(up3)
        up3 = self.convup3_2(up3)
        up3 = self.reluup3_2(up3)       
        # Up 2
        up2 = self.maxpollup2(up3)

        up2 = torch.cat([up2, down2], dim=1) # Skip connection concatenation along the channel axis
        up2 = self.convup2_1(up2)
        up2 = self.reluup2_1(up2)
        up2 = self.convup2_2(up2)
        up2 = self.reluup2_2(up2)   
        # Up 1
        up1 = self.maxpollup1(up2)

        up1 = torch.cat([up1, down1], dim=1) # Skip connection concatenation along the channel axis
        up1 = self.convup1_1(up1)
        up1 = self.reluup1_1(up1)
        up1 = self.convup1_2(up1)
        up1 = self.reluup1_2(up1)  


        # Output ___________________________________________________________________________________
        # Out
        output = self.output_conv(up1) 

        return output       