# 卷积神经网络实践

> 本节介绍如何构造一个简单的CNN模型进行手写数字识别，

> 但在现实场景中，往往使用imagenet预训练的深度CNN模型进行迁移学习，能极大地提升预测准确率，

> 可参考我在百度大数据竞赛中开源的模型: [keras-dog](https://github.com/ahangchen/keras-dogs)

## 数据处理
- dataset处理成四维的，label仍然作为one-hot encoding
```python
def reformat(dataset, labels, image_size, num_labels, num_channels):
    dataset = dataset.reshape(
        (-1, image_size, image_size, num_channels)).astype(np.float32)
    labels = (np.arange(num_labels) == labels[:, None]).astype(np.float32)
    return dataset, labels
```
- 将lesson2的dnn转为cnn很简单，只要把WX+b改为conv2d(X)+b即可
- 关键在于conv2d
- - - 

### `tf.nn.conv2d(input, filter, strides, padding, use_cudnn_on_gpu=None, data_format=None, name=None)` {#conv2d}

给定四维的`input`和`filter` tensor，计算一个二维卷积

##### Args:


*  <b>`input`</b>: A `Tensor`. type必须是以下几种类型之一: `half`, `float32`, `float64`.
*  <b>`filter`</b>: A `Tensor`. type和`input`必须相同
*  <b>`strides`</b>: A list of `ints`.一维，长度4， 在`input`上切片采样时，每个方向上的滑窗步长，必须和format指定的维度同阶
*  <b>`padding`</b>: A `string` from: `"SAME", "VALID"`. padding 算法的类型
*  <b>`use_cudnn_on_gpu`</b>: An optional `bool`. Defaults to `True`.
*  <b>`data_format`</b>: An optional `string` from: `"NHWC", "NCHW"`， 默认为`"NHWC"`。
    指定输入输出数据格式，默认格式为"NHWC", 数据按这样的顺序存储：
        `[batch, in_height, in_width, in_channels]`
    也可以用这种方式："NCHW", 数据按这样的顺序存储：
        `[batch, in_channels, in_height, in_width]`
*  <b>`name`</b>: 操作名，可选.

##### Returns:

  A `Tensor`. type与`input`相同

Given an input tensor of shape `[batch, in_height, in_width, in_channels]`
and a filter / kernel tensor of shape
`[filter_height, filter_width, in_channels, out_channels]`

conv2d实际上执行了以下操作：

1. 将filter转为二维矩阵，shape为
   `[filter_height * filter_width * in_channels, output_channels]`.
2. 从input tensor中提取image patches，每个patch是一个*virtual* tensor，shape`[batch, out_height, out_width,
   filter_height * filter_width * in_channels]`.
3. 将每个filter矩阵和image patch向量相乘

具体来讲，当data_format为NHWC时：

    output[b, i, j, k] =
        sum_{di, dj, q} input[b, strides[1] * i + di, strides[2] * j + dj, q] *
                        filter[di, dj, q, k]

input 中的每个patch都作用于filter，每个patch都能获得其他patch对filter的训练
需要满足`strides[0] = strides[3] = 1`.  大多数水平步长和垂直步长相同的情况下：`strides = [1, stride, stride, 1]`.
- - -

- 然后再接一个WX+b连Relu连WX+b的全连接神经网络即可

## Max Pooling
在tf.nn.conv2d后面接tf.nn.max_pool，将卷积层输出减小，从而减少要调整的参数

### `tf.nn.max_pool(value, ksize, strides, padding, data_format='NHWC', name=None)` {#max_pool}

Performs the max pooling on the input.

##### Args:


*  <b>`value`</b>: A 4-D `Tensor` with shape `[batch, height, width, channels]` and
    type `tf.float32`.
*  <b>`ksize`</b>: A list of ints that has length >= 4.  要执行取最值的切片在各个维度上的尺寸
*  <b>`strides`</b>: A list of ints that has length >= 4.  取切片的步长
*  <b>`padding`</b>: A string, either `'VALID'` or `'SAME'`. padding算法
*  <b>`data_format`</b>: A string. 'NHWC' and 'NCHW' are supported.
*  <b>`name`</b>: 操作名，可选

##### Returns:

  A `Tensor` with type `tf.float32`.  The max pooled output tensor.

- - -

## 优化
仿照lesson2，添加learning rate decay 和 drop out，可以将准确率提高到90.6%

## 补充
- 最近在用GPU版本的TensorFlow，发现，如果import tensorflow放在代码第一行，运行会报段错误（pycharm debug模式下不会），因此最好在import tensorflow前import numpy或者其他的module

## 参考链接
- [Tensorflow 中 conv2d 都干了啥](http://stackoverflow.com/questions/34619177/what-does-tf-nn-conv2d-do-in-tensorflow)
- [TensorFlow Example](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/convolutional_network.py)
