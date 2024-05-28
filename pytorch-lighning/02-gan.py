"""
author: rohan singh
python script for generative adversarial network (gan) using pytorch lightning
"""



# imports
import os
from torch import optim, nn, utils, Tensor
from torchvision import transforms
from torchvision import datasets
import lightning as pl
import torch
from torch.utils.data import random_split, DataLoader
import torch.nn.functional as F
import torchvision
from collections import OrderedDict



# Class for the data Module
class MNISTDataModule(pl.LightningDataModule):


    def __init__(self, data_dir="./data", batch_size=128, num_workers=int(os.cpu_count() / 2)):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1,), (0.3,))
        ])
        
        self.dl_dict = {'batch_size': self.batch_size, 'num_workers': self.num_workers}


    def prepare_data(self):
        datasets.MNIST(self.data_dir, train=True, download=True)
        datasets.MNIST(self.data_dir, train=False, download=True)


    def setup(self, stage=None):
        # Validation data not strictly necessary for GAN but added for completeness
        if stage == "fit" or stage is None:
            mnist_full = datasets.MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

        if stage == "test" or stage is None:
            self.mnist_test = datasets.MNIST(self.data_dir, train=False, transform=self.transform)
        
    
    def train_dataloader(self):
        return DataLoader(self.mnist_train, **self.dl_dict)


    def val_dataloader(self):
        return DataLoader(self.mnist_train, **self.dl_dict)


    def test_dataloader(self):
        return DataLoader(self.mnist_train, **self.dl_dict)
    


# class for the discriminator
class Discriminator(nn.Module):


    def __init__(self):
        super().__init__()
        # Simple CNN
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 1)


    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        # Flatten the tensor so it can be fed into the FC layers
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.sigmoid(x)



# class for the generator
class Generator(nn.Module):


    def __init__(self, latent_dim):
        super().__init__()
        self.lin1 = nn.Linear(latent_dim, 7*7*64)
        self.ct1 = nn.ConvTranspose2d(64, 32, 4, stride=2)
        self.ct2 = nn.ConvTranspose2d(32, 16, 4, stride=2)
        self.conv = nn.Conv2d(16, 1, kernel_size=7)


    def forward(self, x):
        # Pass latent space input into linear layer and reshape
        x = self.lin1(x)
        x = F.relu(x)
        x = x.view(-1, 64, 7, 7)

        # Transposed convolution to 16x16 (64 feature maps)
        x = self.ct1(x)
        x = F.relu(x)
        
        # Transposed convolution to 34x34 (16 feature maps)
        x = self.ct2(x)
        x = F.relu(x)
        
        # Convolution to 28x28 (1 feature map)
        return self.conv(x)



# class for the generative adversarial network

class GAN(pl.LightningModule):


    ## Initialize. Define latent dim, learning rate, and Adam betas 
    def __init__(self, latent_dim=100, lr=0.0002, 

                 b1=0.5, b2=0.999, batch_size=128):
        super().__init__()
        self.save_hyperparameters()

        self.generator = Generator(latent_dim=self.hparams.latent_dim)
        self.discriminator = Discriminator()

        self.validation_z = torch.randn(8, self.hparams.latent_dim)

    

    def forward(self, z):
        return self.generator(z)
    

    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)
    

    def training_step(self, batch, batch_idx, optimizer_idx):
        real_imgs, _ = batch

        # sample noise
        z = torch.randn(real_imgs.shape[0], self.hparams.latent_dim)

        # train generator
        if optimizer_idx == 0:
            self.generated_imgs = self(z)
            predictions = self.discriminator(self.generated_imgs)
            g_loss = self.adversarial_loss(predictions, torch.ones(real_imgs.size(0), 1))
            

            # log sampled images
            sample_imgs = self.generated_imgs[:6]
            grid = torchvision.utils.make_grid(sample_imgs)
            self.logger.experiment.add_image("generated_images", grid, 0)


            tqdm_dict = {"g_loss": g_loss}
            output = OrderedDict({"loss": g_loss, "progress_bar": tqdm_dict, "log": tqdm_dict})
            return output

        # train discriminator
        if optimizer_idx == 1:
            real_preds = self.discriminator(real_imgs)
            real_loss = self.adversarial_loss(real_preds, torch.ones(real_imgs.size(0), 1))

            fake_preds = self.discriminator(self(z).detach())
            fake_loss = self.adversarial_loss(fake_preds, torch.zeros(real_imgs.size(0), 1)) 


            d_loss = (real_loss + fake_loss) / 2
            tqdm_dict = {"d_loss": d_loss}
            output = OrderedDict({"loss": d_loss, "progress_bar": tqdm_dict, "log": tqdm_dict})
            return output
        

    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))
        return [opt_g, opt_d], []
    

    def on_epoch_end(self):
        # log sampled images
        sample_imgs = self(self.validation_z)
        grid = torchvision.utils.make_grid(sample_imgs)
        self.logger.experiment.add_image("generated_images", grid, self.current_epoch)

    

# main function to run the lihgtning trainer
def main():
    trainer = pl.Trainer(max_epochs=20)
    dm = MNISTDataModule()
    model = GAN()
    trainer.fit(model, dm)



if __name__ == "__main__":
    main()


