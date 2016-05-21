# 全连接神经网络
辅助阅读：[TensorFlow中文社区教程](http://www.tensorfly.cn/tfdoc/tutorials/mnist_tf.html) - [英文官方教程](https://www.tensorflow.org/versions/r0.8/tutorials/mnist/pros/index.html#train-the-model)
  
> 代码见：[full_connect.py](../../src/sgd/full_connect.py)

## Linear Model
- 加载lesson 1中的数据集
- 将Data降维成一维，将label映射为one-hot encoding
```python
def reformat(dataset, labels):
    dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
    # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
    labels = (np.arange(num_labels) == labels[:, None]).astype(np.float32)
    return dataset, labels
```
- 使用梯度计算train_loss，用tf.Graph()创建一个级逻辑回归计算单元

  
  - 用tf.constant将dataset和label转为tensorflow可用的训练格式（训练中不可修改）
  ```python
  tf_train_dataset = tf.constant(train_dataset[:train_subset, :])
  ```
  - 用tf.truncated_normal生成正太分布的数据，作为W的初始值，初始化b为0向量
  ```python
  tf.truncated_normal([image_size * image_size, num_labels])
  ```
  - 用tf.variable将上面的矩阵和向量转为tensorflow可用的训练格式（训练中可以修改）
  ```python
  biases = tf.Variable(tf.zeros([num_labels]))
  ```
  - 用tf.matmul实现矩阵相乘，计算WX+b，这里实际上logit只是一个变量，而非结果
  ```python
  tf.matmul(tf_train_dataset, weights)
  ```
  - 用tf.nn.softmax_cross_entropy_with_logits计算WX+b的结果相较于原来的label的train_loss，并求均值，train_loss只是一个变量而非结果
  ```python
  loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))
  ```
  - 使用梯度找到最小train_loss
  ```python
  optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
  ```
  - 计算相对valid_dataset和test_dataset对应的label的train_loss
  
- 重复计算单元反复训练800次，提高其准确度
  - 为了快速查看训练效果，每轮训练只给10000个训练数据(subset)，恩，每次都是相同的训练数据
  - 将计算单元graph传给session
  - 初始化参数
  - 传给session优化器 - train_loss的梯度optimizer，训练损失 - train_loss，每次的预测结果，循环执行训练
  ```python
  with tf.Session(graph=graph) as session:
        tf.initialize_all_variables().run()
        for step in range(num_steps):
            _, l, predictions = session.run([optimizer, loss, train_prediction])
  ```
  - 在循环过程中，W和b会保留，并不断得到修正
  - 在每100次循环后，会用验证集进行验证一次，验证也同时修正了一部分参数
  ```python
  valid_prediction.eval()
  ```
  - 最后用测试集进行测试
  - 注意如果lesson 1中没有对数据进行乱序化，可能训练集预测准确度很高，验证集和测试集准确度会很低
  
  > 这样训练的准确度为83.2%
  
- 使用SGD，即每次只取一小部分数据做训练，计算loss时，也只取一小部分数据计算loss
  - 对应到程序中，即修改计算单元中的训练数据，
    - 每次输入的训练数据只有128个，随机取起点，取连续128个数据：
  ```python
  offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
  batch_data = train_dataset[offset:(offset + batch_size), :]
  batch_labels = train_labels[offset:(offset + batch_size), :]
  ```
  - 由于这里的数据变化，因此用tf.placeholder来存放这块空间
  ```python
  tf_train_dataset = tf.placeholder(tf.float32,
                                          shape=(batch_size, image_size * image_size))
  tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
  ```
  - 计算3000次，训练总数据量为384000，比之前8000000少
  
  > 准确率提高到86.5%
  
- 上面SGD的模型只有一层WX+b，现在使用一个RELU作为中间的隐藏层，连接两个WX+b
  - 仍然只需要修改Graph计算单元为
  ```python
      Y = W2 * RELU(W1*X + b1) + b2
  ```  
  - 为了在数学上满足矩阵运算，我们需要这样的矩阵运算：
  ```
      [10 * 1] = [10 * N] · RELU([N * 784] · [784 * 1] + [N * 1]) + [10 * 1]  
  ```
  - 这里N取1024，即1024个隐藏结点
  - 于是四个参数被修改
  ```python
  weights1 = tf.Variable(
            tf.truncated_normal([image_size * image_size, hidden_node_count]))
  biases1 = tf.Variable(tf.zeros([hidden_node_count]))
  weights2 = tf.Variable(
            tf.truncated_normal([hidden_node_count, num_labels]))
  biases2 = tf.Variable(tf.zeros([num_labels]))
  ```
  - 预测值计算方法改为
  ```python
  ys = tf.matmul(tf_train_dataset, weights1) + biases1
  hidden = tf.nn.relu(ys)
  logits = tf.matmul(hidden, weights2) + biases2
  ```
  - 计算3000次，可以发现准确率提高得很快，最终测试准确率提高到88.8%