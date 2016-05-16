# Stochastic Optimization

- 实践中大量机器学习都是通过梯度算子来求优化的
- 但有一些问题，最大的问题就是，梯度很难计算
- 我们要计算train loss，这需要基于整个数据集的数据做一个计算
- 而计算使 train loss 下降最快的调整方向需要的时间是计算train loss本身的三倍

![](../../res/hard_scale_gradient.png)

- 因此有了SGD：Stochastic Gradient Descent
  - 计算train loss时，只随机取一小部分数据集做为输入
  - 调整W和b时，调整的大小step需要比较小，因为数据集小，我们找到的不一定是对的方向
  - 这样也就增加了调整的次数
  - 但可观地减小了计算量

![](../../res/sgd.png)

## SGD的优化

> 实际上SGD会使得每次寻找的方向都不是很准，因此有了这些优化

- 随机的初始值
![](../../res/init_for_sdg.png)
- Momentum

> 考虑以前的平均调整方向来决定每一步的调整方向
![](../../res/momentum1.jpg)
![](../../res/momentum2.jpg)

- Learning Rate Decay
  - 训练越靠近目标，步长应该越小
  
- Parameter Hyperspace
  - Learning Rate（即调整的step）不是越大越好，可能有瓶颈
  - SGD有许多参数可以调整，所以被称为黑魔法
  ![](../../res/SDG_param.png)
  - AdaGurad
    - 自动执行momentum和learning rate decay
    - 使得SGD对参数不像原来那样敏感
    - 自动调整效果不如原来的好，但仍然是一个option
 