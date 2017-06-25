## skflow
sklearn风格的api，用tensorflow来处理训练sklearn的数据集，现在已经合并到tf.contrib里。

```python
# tensorflow 0.8
import skflow
from sklearn import datasets, metrics

iris = datasets.load_iris()
classifier = skflow.TensorFlowDNNClassifier(hidden_units=[10, 20, 10], n_classes=3)
classifier.fit(iris.data, iris.target)
score = metrics.accuracy_score(iris.target, classifier.predict(iris.data))
print("Accuracy: %f" % score)
```
