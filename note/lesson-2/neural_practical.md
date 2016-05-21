# 全连接神经网络
- 加载lesson 1中的数据集
- 将Data降维成一维，将label映射为one-hot encoding
- 使用梯度计算train_loss，用tf.Graph()创建一个级逻辑回归计算单元
  > 代码见：[full_connect.py](../../src/sgd/full_connect.py)中tf_logist()
  - 用tf.constant将dataset和label转为tensorflow可用的训练格式（训练中不可修改）
  - 用tf.truncated_normal生成正太分布的数据，作为W的初始值，初始化b为0向量
  - 用tf.variable将上面的矩阵和向量转为tensorflow可用的训练格式（训练中可以修改）
  - 用tf.matmul实现矩阵相乘，计算WX+b，这里实际上logit只是一个变量，而非结果
  - 用tf.nn.softmax_cross_entropy_with_logits计算WX+b的结果相较于原来的label的train_loss，train_loss只是一个变量而非结果
  - 用tf.train.GradientDescentOptimizer(0.5).minimize(loss)，使用梯度找到最小train_loss，在这个过程中就调整了W和b
  - 计算相对valid_dataset和test_dataset对应的label的train_loss
  - 为了快速查看训练效果，每轮训练只给10000个训练数据，恩，每次都是相同的训练数据
- 重复计算单元反复训练800次，提高其准确度
  - 将计算单元graph传给session
  - 初始化参数
  - 传给session优化器，即train_loss的梯度optimizer，训练损失train_loss，每次的预测结果
  - 在这个过程中，W和b会保留，并不断得到修正
  - 在每个循环中，会用验证集进行验证，验证也同时修正了一部分参数
  - 最后用测试集进行测试
  - 注意如果lesson 1中没有对数据进行乱序化，可能训练集预测准确度很高，验证集和测试集准确度会很低
  - 这样训练的准确度为83.2%
- 使用SGD，即每次只取一小部分数据做训练，计算loss时，也只取一小部分数据计算loss
  > 代码见：[full_connect.py](../../src/sgd/full_connect.py)中tf_sgd()
  - 对应到程序中，即修改计算单元中的训练数据，
    - 每次输入的训练数据只有128个，随机取起点：offset = (step \* batch_size) % (train_labels.shape[0] - batch_size)
  - 计算3000次，训练总数据量为384000，比之前8000000少，准确率提高到86.5%
- 上面SGD的模型只有一层WX+b，现在使用一个RELU作为中间的隐藏层，连接两个WX+b
  > 代码见：[full_connect.py](../../src/sgd/full_connect.py)中tf_sgd_relu_nn()
  - 仍然只需要修改Graph计算单元为
  ```python
      Y = W2 * RELU(W1*X + b1) + b2
  ```  
  - 为了在数学上满足矩阵运算，我们需要这样的矩阵运算：
  ```
      [10 * 1] = [10 * N] · RELU([N * 784] · [784 * 1] + [N * 1]) + [10 * 1]  
  ```
  - 这里N取1024，即1024个隐藏结点
  - 计算3000次，可以发现准确率提高得很快，最终测试准确率提高到88.8%