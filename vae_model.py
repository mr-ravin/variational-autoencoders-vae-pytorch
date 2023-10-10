import torch
import torch.nn as nn 
import torch.nn.functional as F

class VAE(nn.Module): # input image size: 3 x 128 x 128, here 3 represents number of channels i.e. r,g,b
    def __init__(self, enc_input_channels=3, enc_output_channels=32, latent_dim=16, mode="analysis"):
        super(VAE,self).__init__()
        self.mode = mode
        self.encoder_conv1 = nn.Conv2d(enc_input_channels, enc_output_channels, kernel_size=5,stride=3)
        self.encoder_conv2 = nn.Conv2d(enc_output_channels, enc_output_channels*2, kernel_size=4,stride=2)
        self.encoder_conv3 = nn.Conv2d(enc_output_channels*2, enc_output_channels*4, kernel_size=4,stride=2)
        self.encoder_conv4 = nn.Conv2d(enc_output_channels*4, enc_output_channels*8, kernel_size=3,stride=2)
        self.encoder_conv5 = nn.Conv2d(enc_output_channels*8, enc_output_channels*8, kernel_size=4,stride=1)
        self.fc1_0 = nn.Linear(enc_output_channels*8, 64)
        self.fc1_1 = nn.Linear(64, 64)
        self.fc_mean = nn.Linear(64, latent_dim)
        self.fc_log_var = nn.Linear(64, latent_dim)
        self.fc2_0 = nn.Linear(latent_dim, 64)
        self.fc2_1 = nn.Linear(64, 64)
        self.fc2_2 = nn.Linear(64, enc_output_channels*8)
        self.decoder_deconv5 = nn.ConvTranspose2d(
            in_channels=enc_output_channels*8, out_channels=enc_output_channels*8, kernel_size=4, 
            stride=1,
        )
        self.decoder_deconv4 = nn.ConvTranspose2d(
            in_channels=enc_output_channels*8, out_channels=enc_output_channels*4, kernel_size=3, 
            stride=2,
        )

        self.decoder_deconv3 = nn.ConvTranspose2d(
            in_channels=enc_output_channels*4, out_channels=enc_output_channels*2, kernel_size=4, 
            stride=2,
        )
        self.decoder_deconv2 = nn.ConvTranspose2d(
            in_channels=enc_output_channels*2, out_channels=enc_output_channels, kernel_size=4, 
            stride=2,
        )
        self.decoder_deconv1 = nn.ConvTranspose2d(
            in_channels=enc_output_channels, out_channels=enc_input_channels, kernel_size=5, 
            stride=3,
        )
    
    def reparameterize(self, mu, log_var):
        """
        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """
        std = torch.exp(0.5*log_var) # standard deviation
        eps = torch.randn_like(std) # `randn_like` as we need the same size
        sample = mu + (eps * std) # sampling
        return sample

    def forward(self,x=None, mean=None, log_var=None, mode="train"):
        if mode =="train": # for Training and Validation these values are set to None, while for Image Generation, we will pass mean, and log_var
            if self.mode=="analysis":
                print("input conv1: ",x.shape)
            x = F.relu(self.encoder_conv1(x))
            if self.mode=="analysis":
                print("output conv1: ",x.shape)
                print("input conv2: ",x.shape)
            x = F.relu(self.encoder_conv2(x))
            if self.mode=="analysis":
                print("output conv2: ",x.shape)
                print("input conv3: ",x.shape)
            x = F.relu(self.encoder_conv3(x))
            if self.mode=="analysis":
                print("output conv3: ",x.shape)
                print("input conv4: ",x.shape)
            x = F.relu(self.encoder_conv4(x))
            if self.mode=="analysis":
                print("output conv4: ",x.shape)
                print("input conv5: ",x.shape)
            x = F.relu(self.encoder_conv5(x))
            if self.mode=="analysis":
                print("output conv5: ",x.shape)
                print("########")
            batch, channel, row, col = x.shape
            x = x.view(batch, -1)
            if self.mode=="analysis":
                print(batch, channel, row, col)
                print("size after flatten: ", x.shape)
            hidden = self.fc1_0(x)
            x = F.relu(hidden)
            hidden = self.fc1_1(x)
            # get `mu` and `log_var`
            mean = self.fc_mean(hidden)
            log_var = self.fc_log_var(hidden)
        else:
            batch,_ = mean.shape
            channel, row, col = 256, 1, 1
        z = self.reparameterize(mean, log_var)
        z = self.fc2_0(z)
        z = F.relu(z)
        z = self.fc2_1(z)
        z = F.relu(z)
        z = self.fc2_2(z)
        z = z.view(batch, channel, row, col)

        if self.mode=="analysis":
            print("########")
            # # decoding
        if self.mode=="analysis":
            print("input dconv5: ",z.shape)
        x = F.relu(self.decoder_deconv5(z))
        if self.mode=="analysis":
            print("output dconv5: ",x.shape)
            print("input dconv4: ",x.shape)
        x = F.relu(self.decoder_deconv4(x))
        if self.mode=="analysis":
            print("output dconv4: ",x.shape)
            print("input dconv3: ",x.shape)
        x = F.relu(self.decoder_deconv3(x))
        if self.mode=="analysis":
            print("output dconv3: ",x.shape)
            print("input dconv2: ",x.shape)
        x = F.relu(self.decoder_deconv2(x))
        if self.mode=="analysis":
            print("output dconv2: ",x.shape)
            print("input dconv1: ",x.shape)
        x = F.relu(self.decoder_deconv1(x))
        if self.mode=="analysis":
            print("output dconv1: ",x.shape)
        reconstruction = torch.sigmoid(x)
        #print("output: ",reconstruction.shape)
        if mode=="generate":
            return reconstruction
        else:
            return reconstruction, mean, log_var, z