import numpy as np
import time
from collections import Counter


# kd-tree每个结点中主要包含的数据结构如下
class Node():
    def __init__(self, data, label, depth=0, lchild=None, rchild=None):
        self.data, self.label, self.depth, self.lchild, self.rchild = \
            (data, label, depth, lchild, rchild)


class KdTree():
    def __init__(self, X, y):
        # 拼接数据信息
        y = y[np.newaxis, :]
        self.data = np.hstack([X, y.T])

        self.rootNode = self.buildTree(self.data)

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

    def preOrder(self, node):
        if node is None:
            return
        print("(", node.data, node.label, ')', ':', node.depth)
        self.preOrder(node.lchild)
        self.preOrder(node.rchild)

    def search(self, x, count=1):
        nearest = []
        assert count >= 1 and count <= len(self.data), '错误的k近邻值'
        # 统一状态，初始化nearest中的点
        # for i in range(count):
        #     nearest.append([-1, None]) # 距离， node
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


class KNNKdTree():
    def __init__(self, n_neighbors=3, p=2):
        self.k = n_neighbors

    def fit(self, X_train, y_train):
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)
        self.kdTree = KdTree(self.X_train, self.y_train)

    def predict(self, x):
        nearest, label = self.kdTree.search(x, self.k)
        # print("nearest", [node.data for node in nearest])
        return nearest, label

    def score(self, X_test, y_test):
        right_count = 0
        for x, y in zip(X_test, y_test):
            _, y_pred = self.predict(x)
            if y_pred == y:
                right_count += 1
            print(y_pred)
        return right_count / len(X_test)


def main():
    # 书本 P55 图 3.3 示例
    X_train = np.array([
        [2, 3],
        [5, 4],
        [9, 6],
        [4, 7],
        [8, 1],
        [7, 2]
    ])
    y_train = np.array([1, 1, -1, 1, -1, 1])
    knn_kdTree = KNNKdTree(n_neighbors=3, p=2)
    knn_kdTree.fit(X_train, y_train)
    X_test = np.array([
        [2, 7],
        [3, 3],
        [9, 3],
        [6, 7],
        [8, 3],
        [7, 4]
    ])
    y_test = np.array([1, 1, -1, 1, -1, 1])
    score = knn_kdTree.score(X_test, y_test)
    print('score = ', score)


if __name__ == "__main__":
    main()
