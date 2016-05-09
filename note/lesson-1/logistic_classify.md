# Logistic Classification
> simple but important classifier
     
About

- train your first simple model entirely end to end
- download and pre-process some images for classification
- run an actual logistic classifier on images data
- connect bit of math and code

Detail
- linear classifier

![](../../res/logistic.png)

> 之所以这样建模，是因为线性公式是最简单的数学模型，仅此而已。

- Input: X (e.g. the pixels in an image)
- apply a linear function to X
  - Giant matrix multiply 
  - take inputs as a big vector 
  - multiply input vector with a matrix, W means weights
  - b means biased term
  - machine learning 调整 weights and bias以达到最好的预测效果
- Output: Y, predictions for per output class
  - Y is a vector, represents the probability of each label
  - 好的预测中，正确的label的概率应当更接近1
  - 往往得到的Y一开始不是概率，而是一些具体值（scores/logits），所以需要转换，by：
  
  > Softmax回归模型：[Wikipedia](http://ufldl.stanford.edu/wiki/index.php/Softmax%E5%9B%9E%E5%BD%92) 
  
  ![](../../res/softmax.png)
- Softmax  
  - 代码 [soft_max.py](../../src/soft_max.py)：Softmax实现与应用
  - input的score差异越大（可以全部乘10试试），则输出差异越大，反之差异越小