import matplotlib.pyplot as plt
import numpy as np
from torch import nn, optim
from torch.autograd import Variable
 def test_network(net, trainloader):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    dataiter = iter(trainloader)
    images, labels = dataiter.next()
    inputs = Variable(images)
    targets = Variable(images)
    optimizer.zero_grad()
    output = net.forward(inputs)
    loss = criterion(output, targets)
    loss.backward()
    optimizer.step()
     return True
 def imshow(image, ax=None, title=None, normalize=True):
    if ax is None:
        fig, ax = plt.subplots()
    image = image.numpy().transpose((1, 2, 0))
    
    if normalize:
        mean = np.array([0.485, 0.456, 0.406])
        sd = np.array([0.229, 0.224, 0.225])
        image = sd * image + mean
        image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='both', length=0)
    ax.set_xticklabels('')
    ax.set_yticklabels('')
    
    return ax
 def view_recon(img, recon):
    fig, axes = plt.subplots(ncols=2, sharex=True, sharey=True)
    axes[0].imshow(img.numpy().squeeze())
    axes[1].imshow(recon.data.numpy().squeeze())
    for ax in axes:
        ax.axis('off')
        ax.set_adjustable('box-forced')
        
def view_classify(img, ps):
    num_classes = ps.size()[1]
    ps = ps.data.numpy().squeeze()
    fig, (ax1, ax2) = plt.subplots(figsize=(6,7), ncols=2)
    ax1.imshow(img.numpy().squeeze())
    ax1.axis('off')
    ax2.barh(np.arange(num_classes), ps)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(num_classes))
    ax2.set_yticklabels(np.arange(num_classes).astype(int), size='large');
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)