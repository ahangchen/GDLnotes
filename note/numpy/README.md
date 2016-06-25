# numpy笔记
> 机器学习常常需要fake数据，或者进行数据预处理，numpy是python科学计算的一把利器。

- numpy [官方手册](http://docs.scipy.org/doc/numpy-1.10.1/genindex.html)，支持字母检索

常用方法：

- 生成数据：
  - arange： 生成一定范围内的数据
  - ones_like：生成与参数维度相同的数据
  - random模块：随机相关
    - np.random.shuffle：给一个ndarray做洗牌

- 数学计算：
  - exp：自然指数
  - sum：求和
  - [numpy.linalg.norm](http://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.norm.html)：求模
  
- 数据修改：
  - delete：从一个列表中删除
- 数据格式化：
  - vstack：转为纵向向量

  

