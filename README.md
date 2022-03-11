# MobileNet
Implementation of several MobileNet versions

MobileNet

Статья: Andrew G. Howard, Menglong Zhu, Bo Chen, Dmitry Kalenichenko, Weijun Wang, Tobias Weyand, Marco Andreetto, and Hartwig Adam.
Mobilenets: Efficient convolutional neural networks for mobile vision applications. CoRR, abs/1704.04861, 2017.

Ссылка: https://arxiv.org/abs/1704.04861

Цель: снизить число вычислений, ускорить работу нейросети

Идеи:

1) Depthwise Separable Convolution. Используются свёртки, которые сначала работают с каждым каналом независимо, применяя к каждому каналу ровно одно ядро свёртки, а затем применяют обычную свёртку 1*1. Обычные свёртки (не depthwise) имеют вычислительную сложность M * N * K * K * W * H, где M - число входных каналов, N - число выходных каналов, K - линейный размер ядра свёртки, W * H - размер карты признаков. Depthwise свёртки имеют вычислительную сложность M * K * K * W * H + M * N * W * H, где первое слагаемое отвечает за первый этап, а второе - за второй этап. Количество операций уменьшается (пропорционально 1/N + 1/K^2).

2) Width Multiplier и Resolution Multiplier. Помимо свёрток, авторы статьи используют два гиперпараметра - множителя: a и p. Количество каналов M и N в слоях можно снизить, домножив его на параметр a из интервала (0,1]. А пространственное разрешение можно снизить на этапе входного изображения, домножив его размеры на параметр p из интервала (0,1]. Это помогает уменьшить размер нейросети и снизить вычислительную сложность.

________________________________________


MobileNet v2

Статья: Mark Sandler, Andrew G. Howard, Menglong Zhu, Andrey Zhmoginov, and Liang-Chieh Chen. 
Mobilenetv2: Inverted residuals and linear bottlenecks. mobile networks for classification, detection and segmentation. CoRR, abs/1801.04381, 2018.

Ссылка: https://arxiv.org/abs/1801.04381

Идеи:

Linear Bottlenecks, Inverted residuals. В первой версии MobileNet используются блоки 3*3 DWC + BN + ReLU + 1*1 Conv + BN + ReLU. В MobileNetv2 используются основную часть составляют "обратные" residual блоки (также называемые bottleneck-блоками). В каждом таком блоке входная карта признаков пропускается через 1*1 свёртку, которая увеличивает число каналов в t раз (t > 1), после чего идёт активация ReLU6. Следующим шагом идет 3*3 depthwise свёртка (каналы сворачиваются независимо) и снова ReLU6. Третьим шагом снова идёт свёртка 1 * 1, снижающая число каналов в t' раз (t' > 1), после этого используется линейная активация. Наконец, исходная карта признаков объединяется с полученной с помощью skip-connection (которые помогают градиенту не затухать). Таким образом, карта признаков сначала расширяется, пропускается через свёртку, а затем снова сужается. 
Также, для понижения пространственной размерности используются блоки 1*1 Conv + ReLU6 + 3*3 DWC (stride=2) + ReLU6 + 1*1 Conv + LinearAct.

________________________________________

MobileNet v3

Andrew Howard, Mark Sandler, Grace Chu, Liang-Chieh Chen, Bo Chen, Mingxing Tan, Weijun Wang, Yukun Zhu, Ruoming Pang, Vijay Vasudevan, Quoc V. Le, Hartwig Adam.
Searching for MobileNetV3.  CoRR, abs/1905.02244, 2019

Ссылка: https://arxiv.org/abs/1905.02244

Идеи:

1) Redesigning Expensive Layers. Конец MobileNetv2 выглядит следующим образом (опуская BatchNorm слои и слои активаций; в скобках будет указываться число каналов входной карты признаков): 1*1 Conv (160) + 3*3 DWC (960) + 1*1 Conv (960) + 1*1 Conv (320) + AvgPool (1280) + 1*1 Conv (1280). Авторы замечают, что 1*1 свёртка перед AvgPool, увеличивающая число каналов и работающая с пространственной размерностью 7*7, очень вычислительно затратна. Они предлагают поменять её местами с AvgPool, тогда свёртка будет происходить по пространственной размерности 1*1. Также это позволяет им убрать часть слоёв из предыдущего bottleneck-блока. В итоге они получают 1*1 Conv (160) + AvgPool (960) + 1*1 Conv (960) + 1*1 Conv (1280). Также авторы статьи смогли сократить число фильтров свёртки на первом слое с 32 до 16, применяя после них либо ReLU, либо swish-activation.

2) Nonlinearities. В MobileNetv3 авторы используют hard-swish (x * (ReLU6(x + 3)) / 6) функцию активации вместо swish (x * sigmoid (x)). Вторая была предложена в ряде статей до этого как улучшающая точность работы нейросетей. Она оказывается гораздо дешевле с точки зрения вычислений.

3) Squeeze-and-excite. В MobileNetv3 в bottleneck выход 3*3 DWC сначала подаётся в squeeze-and-excite блок, а уже затем в 1*1 свёртку. Squeeze-and-excite блок - это AvgPool + Dense + ReLU + Dense + h-swish + mul (домножение результата на исходную карту признаков).

________________________________________

В файле requirements.txt описаны необходимые к установке библиотеки.

Обучение запускается из файла train.py. При запуске можно задать следующие параметры:
 
model_type (model type: Small or Large)
num_classes (number of classes, default=10)
batch_size (dataloader batch_size, default=2)
num_epoches (number of epoches, default=50)
use_cuda (1 if cuda is used, 0 otherwise, default=0)
use_tensorboard (1 if log metrics with tensorboard, 0 otherwise, default=0)
train_dataset_path (path to the training set. If not specified, cifar-10 is used. default=None)
val_dataset_path (path to the validation set. If not specified, cifar-10 is used. default=None)

Пример запуска:
python3 train.py --batch_size 32

Если заданы train_dataset_path или val_dataset_path, данные для обучения и валидации берутся из этих папок. Они должны быть разбиты на подпапки по "верным" меткам классов.
