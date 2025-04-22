**Neural network architecture and training**\

The network used is a 10-layer Squeeze-and-Excite Network with Batch Normalisation and Mish nonlinearities.\
```python
self.conv_net = nn.Sequential()
        self.conv_net.add_module("Conv 1", nn.Conv2d(1, 200, 3, 1, 1))
        self.conv_net.add_module("SE 1", SE_Block(200))
        self.conv_net.add_module("Batchnorm 1", nn.BatchNorm2d(200))
        self.conv_net.add_module("Conv activation", nn.Mish())
        self.conv_net.add_module("Conv 2", nn.Conv2d(200, 190, 3, 1, 1))
        self.conv_net.add_module("SE 2", SE_Block(190))
        self.conv_net.add_module("Batchnorm 2", nn.BatchNorm2d(190))
        self.conv_net.add_module("Conv activation 2", nn.Mish())
        self.conv_net.add_module("Conv 3", nn.Conv2d(190, 180, 3, 1, 1))
        self.conv_net.add_module("SE 3", SE_Block(180))
        self.conv_net.add_module("Batchnorm 3", nn.BatchNorm2d(180))
        self.conv_net.add_module("Conv activation 3", nn.Mish())
        self.conv_net.add_module("Conv 4", nn.Conv2d(180, 170, 3, 1, 1))
        self.conv_net.add_module("SE 4", SE_Block(170))
        self.conv_net.add_module("Batchnorm 4", nn.BatchNorm2d(170))
        self.conv_net.add_module("Conv activation 4", nn.Mish())
        self.conv_net.add_module("Conv 5", nn.Conv2d(170, 160, 3, 1, 1))
        self.conv_net.add_module("SE 5", SE_Block(160))
        self.conv_net.add_module("Batchnorm 5", nn.BatchNorm2d(160))
        self.conv_net.add_module("Conv activation 5", nn.Mish())
        self.conv_net.add_module("Conv 6", nn.Conv2d(160, 150, 3, 1, 1))
        self.conv_net.add_module("SE 6", SE_Block(150))
        self.conv_net.add_module("Batchnorm 6", nn.BatchNorm2d(150))
        self.conv_net.add_module("Conv activation 6", nn.Mish())
        self.conv_net.add_module("Conv 7", nn.Conv2d(150, 140, 3, 1, 1))
        self.conv_net.add_module("SE 7", SE_Block(140))
        self.conv_net.add_module("Batchnorm 7", nn.BatchNorm2d(140))
        self.conv_net.add_module("Conv activation 7", nn.Mish())
        self.conv_net.add_module("Conv 8", nn.Conv2d(140, 130, 3, 1, 1))
        self.conv_net.add_module("SE 8", SE_Block(130))
        self.conv_net.add_module("Batchnorm 8", nn.BatchNorm2d(130))
        self.conv_net.add_module("Conv activation 8", nn.Mish())
        self.conv_net.add_module("Conv 9", nn.Conv2d(130, 120, 3, 1, 1))
        self.conv_net.add_module("SE 9", SE_Block(120))
        self.conv_net.add_module("Batchnorm 9", nn.BatchNorm2d(120))
        self.conv_net.add_module("Conv activation 9", nn.Mish())
        self.conv_net.add_module("Conv 10", nn.Conv2d(120, 110, 3, 1, 1))
        self.conv_net.add_module("SE 10", SE_Block(110))
        self.conv_net.add_module("Batchnorm 10", nn.BatchNorm2d(110))
        self.conv_net.add_module("Conv activation 10", nn.Mish())
        self.conv_net.add_module("Flattener", nn.Flatten())

        self.mlp = nn.Sequential()
        self.mlp.add_module("Linear 1", nn.Linear(7040, 1))
        self.mlp.add_module("Activation 1", nn.Sigmoid())
```
The network was trained on 33M positions from the Lichess evaluation database and 3M positions from the Lichess puzzle database, with each puzzle analysed for 1 second by Stockfish 16. A batch size of 2048 was used because of hardware limitations. The SOAP optimizer was used with an initial learning rate of 0.01 and this was decayed when no improvement in validation loss was seen.\
\
**Search algorithm**\
The PUCT algorithm was used, with the exact formula being:\
```python
self.value - helperfuncs.factor * (1 - (min(helperfuncs.decay, 1) * time_fraction)) * np.sqrt(np.log(self.parent.visits + 1) / (self.visits + 1)) - (bonus * helperfuncs.factor * (1 - min(helperfuncs.decay, 1) * time_fraction))
```
self.value - Evaluation of the position as seen from the perspective of the side to play, between 0 and 1.\
helperfuncs.factor - Exploration factor.\
helperfuncs.decay - Exploration decay factor.\
self.parent.visits - Visits to the parent node.\
self.visits - Visits to the current node.\
bonus - Quiescent bonus. If the last move leading to the position was a capture, the capture bonus is used. If the last move leading to the position was a check, the check bonus is used.\
time_fraction - Fraction of time from the start of the search to the total time allocated for this move.\
\
Additionally, value smoothing is also employed. 25% of the value found from the newest search is averaged with 75% of the current value of the node during backprogation.\