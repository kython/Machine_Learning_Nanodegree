# 项目 3: 非监督学习
## 创建用户细分

### 安装

这个项目要求使用 **Python 2.7** 并且需要安装下面这些python包：

- [NumPy](http：//www.numpy.org/)
- [Pandas](http：//pandas.pydata.org)
- [scikit-learn](http：//scikit-learn.org/stable/)

你同样需要安装好相应软件使之能够运行[Jupyter Notebook](http://jupyter.org/)。

优达学城推荐学生安装 [Anaconda](https：//www.continuum.io/downloads), 这是一个已经打包好的python发行版，它包含了我们这个项目需要的所有的库和软件。

### 代码

初始代码包含在 `customer_segments.ipynb` 这个notebook文件中。这里面有一些代码已经实现好来帮助你开始项目，但是为了完成项目，你还需要实现附加的功能。

### 运行

在命令行中，确保当前目录为 `customer_segments.ipynb` 文件夹的最顶层（目录包含本 README 文件），运行下列命令：

```jupyter notebook customer_segments.ipynb```

​这会启动 Jupyter Notebook 并把项目文件打开在你的浏览器中。

## 数据

​这个项目的数据包含在 `customers.csv` 文件中。你能在[UCI 机器学习信息库](https://archive.ics.uci.edu/ml/datasets/Wholesale+customers)页面中找到更多信息。

## 审阅建议

### 选取样本数据的技巧

选择样本点的时候选择有明显特点的数据是非常好的，但是直接选取最大值的样本点有可能会取到异常值。如果你想选择3个明显不同的客户，可以借助可视化来进行。下面的代码可以作为参考，这里我们根据客户在不同种类上的花费选择使用热度图。

```python
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing

# 输出高分辨率图像
%config InlineBackend.figure_format = 'retina'

# 选择前10个数据点
data_10 = data[:10]
X = np.array(data_10, dtype=np.float32)

# 特征缩放
min_max_scaler = preprocessing.MinMaxScaler()
X = min_max_scaler.fit_transform(X)

# 绘制heatmaps
plt.figure(figsize=(12, 6))
sns.heatmap(X, annot=True, cmap='Blues', xticklabels=list(data.keys()))
```
进行比较的时候倾向于平均值，中位数这样的特征会更好。

### 聚类算法比较
高斯混合模型 和 K-Means算法的主要区别就在于是否为[软聚类](https://zh.wikipedia.org/wiki/模糊聚类)方法。