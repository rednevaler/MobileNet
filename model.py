import torch.nn as nn


class HardSwish(nn.Module):
    def __init__(self):
        super(HardSwish, self).__init__()
        self.relu6 = nn.ReLU6(True)
    
    def forward(self, inputs):
        return inputs * self.relu6(inputs + 3) / 6

class HardSigmoid(nn.Module):
    def __init__(self):
        super(HardSigmoid, self).__init__()
        self.relu6 = nn.ReLU6(True)
    
    def forward(self, inputs):
        return self.relu6(inputs + 3) / 6


def _make_divisible(value, divisor):
    """
    Function from the official MobileNetV1 repo
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    new_value = max(divisor, int(value + divisor / 2) // divisor * divisor)
    if new_value < 0.9 * value:
        new_value += divisor
    return new_value


class SqueezeAndExcite(nn.Module):
    def __init__(self, channels):
        super(SqueezeAndExcite, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dense1 = nn.Linear(channels, _make_divisible(channels // 4, 8)) # parameter 4 is from original paper
        self.act1 = nn.ReLU(True)
        self.dense2 = nn.Linear(_make_divisible(channels // 4, 8), channels) # parameter 4 is from original paper
        self.act2 = HardSigmoid()
        self.layers = [self.dense1,
                       self.act1,
                       self.dense2,
                       self.act2]
        self.net = nn.Sequential(*self.layers)
    
    def forward(self, inputs):
        batch_size, channels, width, hight = inputs.size()
        outputs = self.pool(inputs).view(batch_size, channels)
        outputs = self.net(outputs)
        outputs = outputs.view(batch_size, channels, 1, 1)
        outputs = inputs * outputs
        return outputs


class Expansion(nn.Module):
    def __init__(self,
                 input_channels,
                 hidden_channels,
                 kernel_size,
                 stride,
                 use_squeeze_and_excite,
                 use_hardswish):
        
        super(Expansion, self).__init__()
        self.layers = []
        
        # If there is no expansion, the first depthwise convolution is omitted
        if input_channels != hidden_channels:
            self.layers.append(nn.Conv2d(in_channels=input_channels,
                                         out_channels=hidden_channels,
                                         kernel_size=1))
            self.layers.append(nn.BatchNorm2d(hidden_channels))
            if use_hardswish:
                self.layers.append(HardSwish())
            else:
                self.layers.append(nn.ReLU(True))
        self.layers.append(nn.Conv2d(in_channels=hidden_channels,
                                  out_channels=hidden_channels,
                                  kernel_size=kernel_size,
                                  stride=stride,
                                  padding=(kernel_size - 1) // 2,
                                  groups=hidden_channels))
        self.layers.append(nn.BatchNorm2d(hidden_channels))
        if use_hardswish:
            self.layers.append(HardSwish())
        else:
            self.layers.append(nn.ReLU(True))
        if use_squeeze_and_excite:
            self.layers.append(SqueezeAndExcite(hidden_channels))
        self.net = nn.Sequential(*self.layers)
    
    def forward(self, inputs):
        return self.net(inputs)


class Squeeze(nn.Module):
    def __init__(self,
                 hidden_channels,
                 output_channels):
        
        super(Squeeze, self).__init__()
        
        self.net = nn.Sequential(nn.Conv2d(in_channels=hidden_channels,
                                           out_channels=output_channels,
                                           kernel_size=1),
                                 nn.BatchNorm2d(output_channels))
    def forward(self, inputs):
        return self.net(inputs)


class Bottleneck(nn.Module):
    def __init__(self,
                 input_channels,
                 hidden_channels,
                 output_channels,
                 kernel_size,
                 stride,
                 use_squeeze_and_excite,
                 use_hardswish):
        
        super(Bottleneck, self).__init__()
        
        self.layers = []
        
        self.net = nn.Sequential(Expansion(input_channels,
                                           hidden_channels,
                                           kernel_size,
                                           stride,
                                           use_squeeze_and_excite,
                                           use_hardswish),
                                 Squeeze(hidden_channels,
                                         output_channels))
                                     
        self.use_residual = False
        if (stride == 1) and (input_channels == output_channels):
            self.use_residual = True

    def forward(self, inputs):
        outputs = self.net(inputs)
        if self.use_residual:
            outputs = inputs + outputs
        return outputs


class MobileNetV3(nn.Module):
    def __init__(self, 
                 mode, 
                 bottlenecks_params, 
                 num_classes=1000):
        
        super(MobileNetV3, self).__init__()
        
        self.bottlenecks_params = bottlenecks_params
        
        if mode not in ['Large', 'Small']:
            raise "Incorrect mode"

        self.layers = [nn.Conv2d(in_channels=3, 
                                 out_channels=16, 
                                 kernel_size=3, 
                                 stride=2, 
                                 padding=1), 
                       nn.BatchNorm2d(16), 
                       HardSwish()]
        
        input_channels = 16
        
        for kernal_size, exp_sizes, output_channels, \
            use_squeeze_and_excite, use_hardswish, \
            stride in self.bottlenecks_params:
            
            output_channels = _make_divisible(output_channels, 8)
            exp_size = _make_divisible(exp_sizes, 8)
            
            self.layers.append(Bottleneck(input_channels, 
                                          exp_size, 
                                          output_channels, 
                                          kernal_size, 
                                          stride, 
                                          use_squeeze_and_excite, 
                                          use_hardswish))
            input_channels = output_channels
            
        output_channels = 1024 if mode == 'Small' else 1280
        
        self.layers += [nn.Conv2d(in_channels=input_channels, 
                                  out_channels=exp_size, 
                                  kernel_size=1, 
                                  bias=False),
                        nn.BatchNorm2d(exp_size),
                        HardSwish(), 
                        nn.AdaptiveAvgPool2d((1, 1)), 
                        nn.Conv2d(in_channels=exp_size, 
                                  out_channels=output_channels, 
                                  kernel_size=1, 
                                  bias=False), 
                        HardSwish(), 
                        nn.Conv2d(in_channels=output_channels, 
                                  out_channels=num_classes, 
                                  kernel_size=1, 
                                  bias=False)]
        self.net = nn.Sequential(*self.layers)

    def forward(self, inputs):
        inputs = self.net(inputs)
        outputs = inputs.view(inputs.size(0), -1)
        return outputs


def MobileNetV3_Large(**kwargs):
    
    '''15 bottlenecks in large net'''
    
    kernal_sizes = [3, 3, 3, 5, 5, 5, 3, 3, 3, 3, 3, 3, 5, 5, 5]
    exp_sizes = [16, 64, 72, 72, 120, 120, 240, 200, 184, 184, 480, 672, 672, 960, 960]
    output_channels = [16, 24, 24, 40, 40, 40, 80, 80, 80, 80, 112, 112, 160, 160, 160]
    use_squeeze_and_excite = [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    use_hardswish = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    strides = [1, 2, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1]
    
    bottlenecks_params = []
    
    for k, e, o, u_se, u_h, s in zip(kernal_sizes, exp_sizes, 
                                     output_channels, 
                                     use_squeeze_and_excite, 
                                     use_hardswish, strides):
        bottlenecks_params.append([k, e, o, u_se, u_h, s])
    
    return MobileNetV3(bottlenecks_params=bottlenecks_params, mode='Large', **kwargs)


def MobileNetV3_Small(**kwargs):
    
    '''11 bottlenecks in small net'''
    
    kernal_sizes = [3, 3, 3, 5, 5, 5, 5, 5, 5, 5, 5]
    exp_sizes = [16, 72, 88, 96, 240, 240, 120, 144, 288, 576, 576]
    output_channels = [16, 24, 24, 40, 40, 40, 48, 48, 96, 96, 96]
    use_squeeze_and_excite = [1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
    use_hardswish = [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
    strides = [2, 2, 1, 2, 1, 1, 1, 1, 2, 1, 1]

    bottlenecks_params = []
    
    for k, e, o, u_se, u_h, s in zip(kernal_sizes, exp_sizes, 
                                     output_channels, 
                                     use_squeeze_and_excite, 
                                     use_hardswish, strides):
        bottlenecks_params.append([k, e, o, u_se, u_h, s])

    return MobileNetV3(bottlenecks_params=bottlenecks_params, mode='Small', **kwargs)
