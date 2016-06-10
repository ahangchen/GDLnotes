# 循环神经网络实践
## 加载数据
- 使用[text8](http://mattmahoney.net/dc/textdata)作为训练的文本数据集

text8中只包含27种字符：小写的从a到z，以及空格符。如果把它打出来，读起来就像是去掉了所有标点的wikipedia。

- 直接调用lesson1中maybe_download下载text8.zip
- 用zipfile读取zip内容为字符串，并拆分成单词list
- 用connections模块统计单词数量并找出最常见的单词


达成随机取数据的目标

## 构造计算单元

```python
embeddings = tf.Variable(
        tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
```

- 构造一个vocabulary_size x embedding_size的矩阵，作为embeddings容器，
- 有vocabulary_size个容量为embedding_size的向量，每个向量代表一个vocabulary，
- 每个向量的中的分量的值都在-1到1之间随机分布

```python
embed = tf.nn.embedding_lookup(embeddings, train_dataset)
```

- 调用tf.nn.embedding_lookup，索引与train_dataset对应的向量，相当于用train_dataset作为一个id，去检索矩阵中与这个id对应的embedding

```python
loss = tf.reduce_mean(
        tf.nn.sampled_softmax_loss(softmax_weights, softmax_biases, embed,
                                   train_labels, num_sampled, vocabulary_size))
```

- 采样计算训练损失

```python
optimizer = tf.train.AdagradOptimizer(1.0).minimize(loss)
```

- 自适应梯度调节器，调节embedding列表的数据，使得偏差最小

- 预测，并用cos值计算预测向量与实际数据的夹角作为预测准确度（相似度）指标

## 传入数据进行训练
- 切割数据用于训练，其中：

```python
data_index = (data_index + 1) % len(data)
```

- 依旧是每次取一部分随机数据传入
  - 等距离截取一小段文本
  - 构造训练集：每个截取窗口的中间位置作为一个train_data
  - 构造标签：每个截取窗口中，除了train_data之外的部分，随机取几个成为一个list，作为label（这里只随机取了一个）
  - 这样就形成了根据目标词汇预测上下文的机制，即Skip-gram
- 训练100001次，每2000次输出这两千次的平均损失
- 每10000次计算相似度，并输出与验证集中的词最接近的词汇列表
- 用tSNE降维呈现词汇接近程度
- 用matplotlib绘制结果

![](../../res/word2vec_res.png)

## CBOW
上面训练的是Skip-gram模型，是根据目标词汇预测上下文，而word2vec还有一种方式，CBOW，根据上下文预测目标词汇。

实际上就是将Skip-gram中的输入输出反过来。

- 修改截取数据的方式
  - 构造标签：每个截取窗口的中间位置作为一个train_label
  - 构造训练集：每个截取窗口中，除了train_label之外的部分，随机取几个成为一个list，作为train_data（这里只随机取了一个）
  - 这样就形成了根据上下文预测目标词汇的机制，即CBOW



## 参考链接
[林洲汉-知乎](https://www.zhihu.com/question/28473843/answer/68797210)
[词向量](http://www.jeyzhang.com/tensorflow-learning-notes-3.html)


