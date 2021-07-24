# K 近邻法

K 近邻 - K Nearest Neighbor - KNN

**一句话描述**：给定一个训练数据集，对于新输入的实例，在训练数据集中找到与之最邻近的k个实例，投票决定输入实例的类别

1、输入数据集 $T = \{ (\mathbf{x_1}, y_1), (\mathbf{x_2}, y_2), \dots, (\mathbf{x_n}, y_n) \}$​​​​ ；其中 $\mathbf{x_i}$​​​​​ 为第 $i$​​ 个数据的特征向量；$y_i \in \mathcal{Y} = \{ c_1, c_2, \dots, c_K \}$​ 为第 $i$​​​​ 个数据的类别，共 $K$ 个类别。

2、根据给定的距离度量，在训练数据集 $T$ 中找出与新输入实例 $\mathbf{z}$ 最邻近的 $k$ 个点，涵盖这 $k$ 个点的 $\mathbf{z}$ 的邻域记作：$N_k(\mathbf{z})$​

3、根据分类决策规则，在 $N_k(\mathbf{z})$​ 中决定输入实例的类别：$\mathop{\arg\max}\limits_{c_j} \sum\limits_{x_i \in N_k(\mathbf{z})} I(y_i = c_j)$​ ；其中 $I$​ 为指示函数，$y_i=c_j$​​ 时值为 1 否则为 0。



根据上述的过程，我们还需要了解下：距离度量，k 值的选择，分类决策规则。此外当训练数据集 $T$​​ 中节点过多时，计算得到 $N_k(\mathbf{z})$​​ 将耗费大量算力，因此需要引入 $kd\ Tree$​​ 这一数据结构来解决问题。

# 距离度量

KNN 中将两个实例点间的距离作为其相似程度的度量，一般的距离度量都选用 $L_p$ 距离
$$
L_p(\mathbf{x}_i, \mathbf{x}_j) = \left( \sum_{l=1}^{n} \left| x_i^{(l)} - x_j^{(l)} \right|^p \right) ^{\frac{1}{p}}
$$
比如当 $p=2$​ 时为我们常见的欧式距离，$p=1$​ 时为曼哈顿距离

<img src="https://gitee.com/Butterflier/pictures/raw/master/image-20210724113727757.png" alt="image-20210724113727757" style="zoom:50%;" />

特别的还应注意当 $p \rightarrow +\infty$  其距离度量是各个维度距离的最大值，同样的当 $p \rightarrow -\infty$ 其距离度量是各个维度距离的最小值
$$
L_{+\infty}(\mathbf{x}_i, \mathbf{x}_j) = \max_l \left| x_i^{(l)} - x_j^{(l)} \right|
$$
有关距离度量的选择可以根据任务需求进行判断，在 KNN 中通常使用欧式距离作为其距离度量。

Tips：为了避免各个维度数量级不同导致距离大小只取决于单一特征的问题出现，通常需要进行特征归一化 / 标准化。例如房价这一特征在 [5000, 10000] 范围中变化，而房间大小却在 [90, 120] 之间变化，那么计算距离时，房间大小这一特征在房价特征前就无法充分表达。

# k 值的选择

**较小的 k 值**：近似误差会减小（只有与输入实例较劲的训练实例才能起作用），估计误差会增大（预测结果会对近邻的实例点非常敏感）。
**较大的 k 值**：与之相反。与输入实例不相似的点也能起到预测作用，使得预测可能发生错误。k 值的增大使得模型简单。
**通常决策**：先选取较小的 k 值，后用交叉验证的方法确定最优 k 值。

# 分类决策规则

在这里 KNN 采用多数表决（Vote），为什么多数表决合理呢？是因为多数表决规则等价于经验风险最小化，证明如下：

假设我们的损失函数为 0,1损失函数 - 分类错误 $loss + 1$，分类函数为 $f(\mathbf{z})$​​

误分类概率：$P(y \neq f(\mathbf{z})) = 1 - P(y=f(\mathbf{z}))$​​

误分类率：$\frac{1}{k} \sum\limits_{x_i \in N_k(\mathbf{z})} I(y_i \ne c_j) = 1 - \frac{1}{k} \sum\limits_{x_i \in N_k(\mathbf{z})} I(y_i = c_j)$​​​ 

因此为了使误分类率最小，就要最大化：$\frac{1}{k} \sum\limits_{x_i \in N_k(\mathbf{z})} I(y_i = c_j)$

而使之最大化，则就应使得 $c_j$​ 为 $N_k(\mathbf{z})$​​  中的大多数表达，即为多数表决规则(Vote)

Tips：多数表决并不和 0,1 损失挂钩，在这里仅仅是一个简单的示例，采用别的损失函数，也能得到同样的结论。

# kd Tree

这才是 KNN 中的重头戏啊~

针对特征空间维度较大以及数据容量较大的情况，使用 kd Tree 可以加速检索过程。

kd Tree 是一个二叉树，目的是实现对 k 维空间的划分。

## 构造方法

基本思路：顺序选定一个轴（维度）切下去，这样实力就会被划分为（左右）两个区域；为了保证树的平衡，选择该轴的中位数为切分点。

直到划分的子区域中没有实例为止。

样例：$T = \{(2,3)^T, (5,4)^T, (9,6)^T, (4,7)^T, (8,1)^T, (7,2)^T\}$​

1、选择第一个维度准备下刀，选择中位数（2, 4, 5, 7, 8, 9)，（中位数应为 6，但是 $x^{(1)}=6$​ 并没有对应数据点，向上向下选择都可以，这里按照书上向上选取），选择 $(7, 2)$ 为切分点。

2、左侧区域为 (2, 3) (5, 4) (4, 7)，选择第二个维度下刀，选择中心点 (5, 4) 开切

2、左侧区域为 (8, 1) (9, 6)，选择第二个维度下刀，选择中心点 (9, 6) 开切

3、(5, 4) 左侧区域为 (2, 3)，选择第一个维度下刀，选择中心点 (2, 3) 开切 —— END

3、(5, 4) 右侧区域为 (4, 7)，选择第一个维度下刀，选择中心点 (4, 7) 开切 —— END

3、(9, 6) 左侧区域为 (8, 1)，选择第一个维度下刀，选择中心点 (8, 1) 开切 —— END

3、(9, 6) 右侧 —— END

下图 3.3 为划分结果，可以简单想下，在我们求最近邻时，向左或向右决策就可以舍去剩余节点一半的点，确实可以加速┗|｀O′|┛ 嗷~~

<img src="https://gitee.com/Butterflier/pictures/raw/master/image-20210724135524379.png" alt="image-20210724135524379" style="zoom:50%;" />

## 搜索方法

对于目标实例 $\mathbf{z}$​ ，我们的目标是利用 kd Tree 搜寻其 $k$ 近邻点

1、定位目标实例到最小子空间：根据每层的划分维度，对 $\mathbf{z}$​ 进行子空间定位，一直定位到原子空间 —— 小于该层的纬度值则向左，大于向右。

2、递归向上执行，维护 $k$ 近邻节点集合 $S$

2.1、如果 $S$​ 未满，直接添加当前子空间节点

2.2、如果近邻集合满，比较距离判断是否替换

3、查看当前节点的另一侧子区域：判断当前近邻集合中距离 $\mathbf{z}$​ 的最大距离是否能跨域 $\mathbf{z}$​​ 所属的子区域（也就是下图所示的情况），我们需要搜索 A 的右侧空间，通过搜索 C 找到 E

> 需要针对当前所在维度看，仅仅看一个维度，因此比较距离就相当于在一个轴上比较
>
> 如果距离超出，判断节点 $\mathbf{z}$​ 在当前空间的左侧还是右侧
>
> 如果在左侧：则向右寻找；如果在右侧：则向左寻找；

![image-20210724145306684](https://gitee.com/Butterflier/pictures/raw/master/image-20210724145306684.png)

## Python 实现

### 节点类

```python
# kd-tree每个结点中主要包含的数据结构如下
# data - 节点的特征向量，label - 节点的标签，depth - 节点的深度（用于判断切分维度），lchild，rchild - 节点指向的左右空间节点
class Node():
    def __init__(self, data, label, depth=0, lchild=None, rchild=None):
        self.data, self.label, self.depth, self.lchild, self.rchild = \
            (data, label, depth, lchild, rchild)
```

### 初始化

```python
def __init__(self, X, y):
    # 拼接数据信息
    y = y[np.newaxis, :]
    self.data = np.hstack([X, y.T])
	# 递归构建
    self.rootNode = self.buildTree(self.data)
```

### 构建

```python
# data - 总体数据信息， depth - 当前构建的深度，用于判断当前划分的维度
def buildTree(self, data, depth=0):
```

1、选择划分维度

```python
# depth 为递归时记录的深度值，初始为 0
m, self.n = np.shape(data) # m 条数据，n 个维度
aim_axis = depth % (self.n-1) # 选择切分的维度
```

2、选择目标维度的中位数

```python
# 排序寻找中位数
sorted_data = sorted(data, key=lambda item: item[aim_axis])
mid = m // 2
```

3、确认当前超空间节点

```python
# 记录该空间对应的节点
node = Node(sorted_data[mid][:-1], sorted_data[mid][-1], depth=depth)
```

4、进行左右空间的划分

```python
node.lchild = self.buildTree(sorted_data[:mid], depth=depth+1)
node.rchild = self.buildTree(sorted_data[mid+1:], depth=depth+1)
```

5、递归完成后，向上返回当前根节点

```python
return node
```

**整体代码**

```python
    def buildTree(self, data, depth=0):
        if(len(data) <= 0):
            return None

        # m 条数据，n 个维度
        m, self.n = np.shape(data)
        # 选择切分的维度
        aim_axis = depth % (self.n-1)

        # 排序寻找中位数
        sorted_data = sorted(data, key=lambda item: item[aim_axis])
        mid = m // 2

        # 切分，并记录该空间对应的节点
        node = Node(sorted_data[mid][:-1], sorted_data[mid][-1], depth=depth)

        # 记录根节点
        if(depth == 0):
            self.kdTree = node

        node.lchild = self.buildTree(sorted_data[:mid], depth=depth+1)
        node.rchild = self.buildTree(sorted_data[mid+1:], depth=depth+1)
        return node
```

### 搜索

向上递归进行搜索，全程维护 nearest - k 近邻节点集合

主要内容为 `recurve(node)` 方法，node 表示当前子空间的管理节点信息

1、首先获取划分维度，一层一层的划分到叶子节点 即 原子空间

```python
# 获取当前深度所划分的维度
now_axis = node.depth % (self.n - 1)

if(x[now_axis] < node.data[now_axis]):
    recurve(node.lchild)
else:
    recurve(node.rchild)
```

2、到达叶子节点，更新 k 近邻节点集合

```python
# 到达了叶子节点，或者子节点判断完毕
dist = np.linalg.norm(x - node.data, ord=2)

# 更新近邻信息
# 近邻节点结合没有满，则直接加入
if(len(self.nearest) < count):
    self.nearest.append([dist, node])
else:
    # 集合满了按照距离大小进行替换
    aim_index = -1
    for i, d in enumerate(self.nearest):
        if(d[0] < 0 or dist < d[0]):
            aim_index = i
    if(aim_index != -1):
        self.nearest[aim_index] = [dist, node]
```

3、判断 $k$​ 近节点是否可能出现在另一侧，即下图点 $E$

![](https://gitee.com/Butterflier/pictures/raw/master/image-20210724145306684.png)

注意当 node 为 D 节点的时候可定不会涉及到 D 的左侧空间 (下侧)，但是当递归到 A 节点进行判断时，会发现 A 的右侧（上侧）空间中可能存在更优节点，因此将会对目标空间进一步搜索。

```python
# 获取当前近邻点中距离最大值的索引
max_index = np.argmax(np.array(self.nearest)[:, 0])
# 表示这个近邻点的距离跨域了当前节点所表示的子区域，所以需要调查另一子树
if(self.nearest[max_index][0] > abs(x[now_axis] - node.data[now_axis])):
	if(x[now_axis] - node.data[now_axis] < 0):
		recurve(node.rchild)
	else:
		recurve(node.lchild)
```

**整体代码**

```python
def search(self, x, count=1):
    nearest = []
    assert count >= 1 and count <= len(self.data), '错误的k近邻值'
    self.nearest = nearest

    def recurve(node):
        if node is None:
            return
        # 获取当前深度所划分的维度
        now_axis = node.depth % (self.n - 1)

        if(x[now_axis] < node.data[now_axis]):
            recurve(node.lchild)
        else:
            recurve(node.rchild)

        # 到达了叶子节点，或者子节点判断完毕
        dist = np.linalg.norm(x - node.data, ord=2)

        # 更新近邻信息
        if(len(self.nearest) < count):
            self.nearest.append([dist, node])
        else:
            aim_index = -1
            for i, d in enumerate(self.nearest):
                if(d[0] < 0 or dist < d[0]):
                    aim_index = i
            if(aim_index != -1):
                self.nearest[aim_index] = [dist, node]
        # 获取当前近邻点中距离最大值的索引
        max_index = np.argmax(np.array(self.nearest)[:, 0])
        # 表示这个近邻点的距离跨域了当前节点所表示的子区域，所以需要调查另一子树
        if(self.nearest[max_index][0] > abs(x[now_axis] - node.data[now_axis])):
            if(x[now_axis] - node.data[now_axis] < 0):
                recurve(node.rchild)
            else:
                recurve(node.lchild)

    recurve(self.rootNode)
    poll = [item[-1].label for item in self.nearest]
    return self.nearest, Counter(poll).most_common()[0][0]
```