import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from logger import Logger


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# MNIST dataset 
dataset = torchvision.datasets.MNIST(root='/home/imc/XR/temp/learning-tensorboard/data',
                                     train=True, 
                                     transform=transforms.ToTensor(),  #要将所有的image变成tensor，所以有个transform
                                     download=False)

# Data loader
'''
我们以前手动加载数据的方式，在数据量小的时候，并没有太大问题，但是到了大数据量，我们需要使用 shuffle, 分割成mini-batch 等操作的时候，我们可以使用PyTorch的API快速地完成这些操作。
DataLoader是一个比较重要的类，它为我们提供的常用操作有：
1、dataset：（数据类型 dataset）
输入的数据类型。看名字感觉就像是数据库，C#里面也有dataset类，理论上应该还有下一级的datatable。这应当是原始数据的输入。PyTorch内也有这种数据结构。这里先不管，估计和C#的类似，这里只需要知道是输入数据类型是dataset就可以了。
2、batch_size：（数据类型 int）
每次输入数据的行数，默认为1。PyTorch训练模型时调用数据不是一行一行进行的（这样太没效率），而是一捆一捆来的。这里就是定义每次喂给神经网络多少行数据，如果设置成1，那就是一行一行进行。
3、shuffle：（数据类型 bool）
洗牌。默认设置为False。在每次迭代训练时是否将数据洗牌，默认设置是False。将输入数据的顺序打乱，是为了使数据更有独立性，但如果数据是有序列特征的，就不要设置成True了。
从DataLoader类的属性定义中可以看出，这个类的作用就是实现数据以什么方式输入到什么网络中。

代码一般是这么写的：
# 定义学习集 DataLoader
train_data = torch.utils.data.DataLoader(各种设置...)

# 将数据喂入神经网络进行训练
for i, (input, target) in enumerate(train_data): 
    循环代码行......
'''
data_loader = torch.utils.data.DataLoader(dataset=dataset, # 在这里要将我们的datasets转变为dataloader
                                          batch_size=100,  # 100个sample为1个batch
                                          shuffle=True)


# Fully connected neural network with one hidden layer
'''
input_size =784     # The image size = 28 x 28 = 784
hidden_size =500    # The number of nodes at the hidden layer
num_classes =10     # The number of output classes. In this case, from 0 to 9
'''
class NeuralNet(nn.Module):   # 定义含有一个隐含层的全连接神经网络。
    def __init__(self, input_size=784, hidden_size=500, num_classes=10):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # 输入层
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)  #  输出层

    #只定义了forward函数，backward函数（计算梯度的位置）通过使用autograd被自动定义了。
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# 定义模型
model = NeuralNet().to(device)  # NeuralNet(input_size, hidden_size, num_classes).to(device)这里的输入尺寸和隐藏尺寸要和训练的图片保持一致的

logger = Logger('logs') #该路径存放我们作图的文件

# Loss and optimizer 损失函数和优化算法
'''
为了训练网络，都需要定义一个loss function来描述模型对问题的求解精度。loss越小，代表模型的结果和真实值偏差越小，这里使用CrossEntropyLoss()来计算．Adam，这是一种基于一阶梯度来优化随机目标函数的算法。
'''
criterion = nn.CrossEntropyLoss()  # 针对单目标分类问题, 结合了 nn.LogSoftmax() 和 nn.NLLLoss() 来计算 loss.
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)  #优化器，设置学习的速度和使用的模型


# 接下来就是训练模型了，训练模型这部分是有点绕的，首先我们来看代码，后面再针对各个函数做说明：
data_iter = iter(data_loader) #迭代器 。有了dataloader之后我们需要一个batch一个batch的调出来，所以要把data_loader变成iterator
iter_per_epoch = len(data_loader) #这里看一下有多少个mini batch
total_step = 50000

# Start training
for step in range(total_step): #在这里我们一个mini-batch一个mini-batch的来进行训练
    
    # Reset the data_iter 这两行的意思是我们每训练完一次全样本，就对我们所有的全样本做一次resize它的iterator，即做的shuffle的工作
    if (step+1) % iter_per_epoch == 0:
        data_iter = iter(data_loader)  #iter就是对全样本一个重新的shuffle

    # Fetch images and labels
    images, labels = next(data_iter)  # 有了iterator之后就通过next的方式就一个batch一个batch的调取出来
    # 有了一个batch的image和label之后要将它从tensor变成variable。同时对image而言要将它的shape进行改变，由原来可能三维四维的数据变成二维
    images, labels = images.view(images.size(0), -1).to(device), labels.to(device)
    
    # Forward pass
    outputs = model(images)
    loss = criterion(outputs, labels) #将输出的outputs和原来导入的labels作为loss函数的输入就可以得到损失了
    
    # Backward and optimize
    optimizer.zero_grad() #训练之前需要把所有参数的导数归零，也就是把loss关于weight的导数变成0.
    loss.backward() # 计算得到loss后就要回传损失。要注意的是这是在训练的时候才会有的操作，测试时候只有forward过程
    #回传损失过程中会计算梯度，然后需要根据这些梯度更新参数，optimizer.step()就是用来更新参数的。optimizer.step()后，
    # 你就可以从optimizer.param_groups[0][‘params’]里面看到各个层的梯度和权值信息。
    optimizer.step() #有了更新好的倒数，我们就可以更新我们的参数。这里是只更新一步



    # Compute accuracy
    _, argmax = torch.max(outputs.data, 1) #计算accuracy之前首先要将output做squeeze 即由两列变成一列之后取最大一列的值，然后通过这种方式取它的index即argmax而不是取值本身
    '''
    训练好的数据怎么和预测联系起来呢？
训练输出的outputs也是torch.autograd.Variable格式，得到输出后（网络的全连接层的输出）还希望能到到模型预测该样本属于哪个类别的信息，这里采用
torch.max。torch.max()的第一个输入是tensor格式，所以用outputs.data而不是outputs作为输入；第二个参数1是代表dim的意思，也就是取每一行的最大值，
其实就是我们常见的取概率最大的那个index；第三个参数loss也是torch.autograd.Variable格式。
    '''
    accuracy = (labels == argmax.squeeze()).float().mean()

    if (step+1) % 100 == 0: #每训练一百次就统计一次画图上的更新
        print ('Step [{}/{}], Loss: {:.4f}, Acc: {:.2f}' 
               .format(step+1, total_step, loss.item(), accuracy.item()))

        # ================================================================== #
        #                        Tensorboard Logging                         #
        # ================================================================== #

        # 1. Log scalar values (scalar summary) 第一个要画的图是损失函数的曲线和accuracy的曲线
        info = { 'loss': loss.item(), 'accuracy': accuracy.item() } #这里只需要将loss和accuracy提供出来就行。注意这里不是tensor也不是numpy array而是单个的scalar

        for tag, value in info.items():  #上面是一个dic，有了这样的dic就可以一个个的带出来再带入到logger.scalar_summary当中去，同时step+1即加上index的序号
            logger.scalar_summary(tag, value, step+1)



        # 2. Log values and gradients of the parameters (histogram summary)#这里是针对所有的parameters和gradient来做histogram
        for tag, value in model.named_parameters():  #基于所有的parameter它的named的iterator把他的名字tag和它的值都带出来
            tag = tag.replace('.', '/')
            #下面这两行也是把名字和值都带进去，index即step也带进去。这里使用的histo_summary的method
            logger.histo_summary(tag, value.data.cpu().numpy(), step+1) #这里把parameter的object变成np.array
            logger.histo_summary(tag+'/grad', value.grad.data.cpu().numpy(), step+1)  #这里把variable变成np.array

        # 3. Log training images (image summary)  最后就是作图，要注意的是要把原来的tensor变成三维的
        info = { 'images': images.view(-1, 28, 28)[:10].cpu().numpy() }  #这里只取了10张图片然后把它放入一个dictornary

        for tag, images in info.items():  #在这里把dic打开，这里有他的index即tag，还有它的image值tensor本身
            logger.image_summary(tag, images, step+1) #这里把他的index和它图片的值一个一个放进去，step+1是batch的序号


        #以上完成了每一百个min-batch就做一次图