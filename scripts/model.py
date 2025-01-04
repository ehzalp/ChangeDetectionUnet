from imports import *

class UNet(nn.Module):
    def __init__(self, n_classes):
        """
        Initializes the U-Net architecture with the specified number of output classes.
        n_classes: Number of output classes (e.g., 1 for binary segmentation)
        """
        super(UNet, self).__init__()
        # TODO: Define needed layers, use n_class variable in the last layer.

        # Define the encoder layers
        self.conv1 = nn.Conv2d(6, 64, kernel_size=3, padding=1)  # Input channels are 6 (3 from img_A and 3 from img_B)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.relu5 = nn.ReLU()
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.relu6 = nn.ReLU()
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv7 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.relu7 = nn.ReLU()
        self.conv8 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu8 = nn.ReLU()

        # Define the decoder layers
        self.upconv1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv9 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.relu9 = nn.ReLU()
        self.conv10 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.relu10 = nn.ReLU()

        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv11 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.relu11 = nn.ReLU()
        self.conv12 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.relu12 = nn.ReLU()

        self.upconv3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv13 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.relu13 = nn.ReLU()
        self.conv14 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.relu14 = nn.ReLU()

        # Define the output layer
        self.conv15 = nn.Conv2d(64, n_classes, kernel_size=1)
        self.out15 = nn.Sigmoid()

    def forward(self, x):
        """
        Defines the forward pass of the U-Net model.
        x: Input tensor (a combination of before and after images)
        Returns:
        out_mask: The predicted segmentation mask after processing through the network
        """
        # Encoder
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out1 = out
        out = self.maxpool1(out)

        out = self.conv3(out)
        out = self.relu3(out)
        out = self.conv4(out)
        out = self.relu4(out)
        out2 = out
        out = self.maxpool2(out)

        out = self.conv5(out)
        out = self.relu5(out)
        out = self.conv6(out)
        out = self.relu6(out)
        out3 = out
        out = self.maxpool3(out)

        # Base convolutional layers
        out = self.conv7(out)
        out = self.relu7(out)
        out = self.conv8(out)
        out = self.relu8(out)

        # Decoder
        out = self.upconv1(out)
        out = torch.cat((out, out3), dim=1)
        out = self.conv9(out)
        out = self.relu9(out)
        out = self.conv10(out)
        out = self.relu10(out)

        out = self.upconv2(out)
        out = torch.cat((out, out2), dim=1)
        out = self.conv11(out)
        out = self.relu11(out)
        out = self.conv12(out)
        out = self.relu12(out)

        out = self.upconv3(out)
        out = torch.cat((out, out1), dim=1)
        out = self.conv13(out)
        out = self.relu13(out)
        out = self.conv14(out)
        out = self.relu14(out)
        out = self.conv15(out)
        out_mask = self.out15(out)

        return out

