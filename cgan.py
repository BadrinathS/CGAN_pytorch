import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.utils import save_image
import wandb

import numpy as np
import matplotlib.pyplot as plt

wandb.init(job_type='train', project='CGAN')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

bs = 100

train_dataset = datasets.MNIST(root='./mnist_data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root='./mnist_data', train=False, transform=transforms.ToTensor(), download=False)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size =bs, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=bs, shuffle=True)

class Generator(nn.Module):
    def __init__(self, input_dim, output_dim, c_dim):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(input_dim + c_dim, 256)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features*2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features*2)
        self.fc4 = nn.Linear(self.fc3.out_features, output_dim)
    
    def forward(self, x, c):
        concat_input = torch.cat([x,c], 1)
        x = F.leaky_relu(self.fc1(concat_input), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = F.leaky_relu(self.fc4(x), 0.2)

        return x
class Discriminator(nn.Module):
    def __init__(self, input_dim, c_dim):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(input_dim+c_dim, 1024)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features//2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features//2)
        self.fc4 = nn.Linear(self.fc3.out_features, 1)
    
    def forward(self, x, c):
        concat_input = torch.cat([x,c], 1)
        x = F.leaky_relu(self.fc1(concat_input), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = torch.sigmoid(self.fc4(x))

        return x

def one_hot_encoding(labels, c_dim):
    targets = torch.zeros(labels.size(0), c_dim)
    
    for i, label in enumerate(labels):
        targets[i,label] = 1
    return targets

g_input_dim = 100
g_output_dim = train_dataset.train_data.size(1)*train_dataset.train_data.size(2)
c_dim = train_loader.dataset.train_labels.unique().size(0)

d_input_dim = train_dataset.train_data.size(1)*train_dataset.train_data.size(2)

generator = Generator(g_input_dim, g_output_dim, c_dim).to(device)
discriminator = Discriminator(d_input_dim, c_dim).to(device)

criterion = nn.BCELoss()

lr = 0.0002
generator_optimizer = optim.Adam(generator.parameters(), lr=lr)
discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=lr)

def train_discriminator(x, label):
    discriminator.zero_grad()

    x_real, y_real = x.view(-1,g_output_dim), torch.ones(bs, 1)
    label_real = Variable(one_hot_encoding(label, c_dim).to(device))
    x_real, y_real = Variable(x_real.to(device)), Variable(y_real.to(device))
    discriminator_output = discriminator(x_real, label_real)
    discriminator_real_loss = torch.log(discriminator_output + 0.0001)

    z = torch.randn(bs, g_input_dim).to(device)
    label_fake = torch.LongTensor(np.random.randint(0,10, bs)).to(device)
    label_fake = one_hot_encoding(label_fake, c_dim).to(device)
    x_fake, y_fake = generator(z, label_fake), torch.zeros(bs, 1).to(device)

    discriminator_output = discriminator(x_fake, label_fake)
    discriminator_fake_loss = torch.log(1-discriminator_output + 0.0001)

    discriminator_loss = -torch.mean(discriminator_real_loss + discriminator_fake_loss)
    discriminator_loss.backward()
    discriminator_optimizer.step()

    return discriminator_loss.item()

def train_generator(x, label):
    generator.zero_grad()

    z = torch.randn(bs, g_input_dim).to(device)
    label_real = one_hot_encoding(label, c_dim).to(device)
    y = torch.ones(bs, 1).to(device)
    generator_output = generator(z, label_real)
    discriminator_output = discriminator(generator_output, label_real)
    generator_loss = -torch.mean(torch.log(discriminator_output + 0.0001)) #Use either of loss function, although this one is easier to optimize

    generator_loss.backward()
    generator_optimizer.step()

    return generator_loss.item()

n_epoch = 50
for epoch in range(n_epoch):

    d_losses, g_losses = [], []


    for batch_id, (x,y) in enumerate(train_loader):
        g_losses.append(train_generator(x, y))
        d_losses.append(train_discriminator(x, y))
        
        
    if epoch % 10 == 0:
        with torch.no_grad():
            test_z = torch.randn(bs, g_input_dim).to(device)
            label_fake = torch.LongTensor(np.random.randint(0,10, bs)).to(device)
            label_fake = one_hot_encoding(label_fake, c_dim).to(device)
            generated_output = generator(test_z, label_fake)
            name = './images/sample_' + str(epoch) + '.png'
            save_image(generated_output.view(bs, 1, 28,28), name)  

    print('Epoch %d \t loss_d %f \t loss_g %f'%( epoch, torch.mean(torch.FloatTensor(d_losses)).item(), torch.mean(torch.FloatTensor(g_losses)).item()))
    wandb.log({'Generator Loss': torch.mean(torch.FloatTensor(g_losses)).item(), 'Discriminator Loss': torch.mean(torch.FloatTensor(d_losses)).item()}, step=epoch)
    

    test_z = torch.randn(bs, g_input_dim).to(device)
    label_fake = torch.LongTensor(np.random.randint(0,10, bs)).to(device)
    label_fake = one_hot_encoding(label_fake, c_dim).to(device)
    generated_output = generator(test_z, label_fake)
    img = generated_output.view(bs, 1, 28,28)
    wandb.log({"Images":[wandb.Image(img, caption="Image for epoch "+str(epoch))]}, step=epoch)


torch.save(generator.state_dict(), './ckpt/generator.pth')
torch.save(discriminator.state_dict(), './ckpt/discriminator.pth')

with torch.no_grad():
    test_z = torch.randn(bs, g_input_dim).to(device)
    label_fake = torch.LongTensor(np.random.randint(0,10, bs)).to(device)
    label_fake = one_hot_encoding(label_fake, c_dim).to(device)
    generated_output = generator(test_z, label_fake)

    save_image(generated_output.view(bs, 1, 28,28), './images/sample_final.png')