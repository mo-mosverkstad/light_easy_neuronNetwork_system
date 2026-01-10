import numpy as np

# 生成玩具数据：两类点
np.random.seed(42)
N = 200  # 样本数
X_class0 = np.random.randn(N//2, 2) + np.array([0.0, -2.0])
X_class1 = np.random.randn(N//2, 2) + np.array([2.5, 2.0])
X = np.vstack([X_class0, X_class1])           # 形状 (N, 2)
y = np.array([0]*(N//2) + [1]*(N//2))         # 标签 0/1
y = y.reshape(-1, 1)

# 标准化（可选）
X = (X - X.mean(axis=0)) / X.std(axis=0)

# 模型结构：2 -> 16 -> 1（Sigmoid）
input_dim = 2
hidden_dim = 16
output_dim = 1
lr = 0.1
epochs = 3000

# 参数初始化
W1 = np.random.randn(input_dim, hidden_dim) * 0.1
b1 = np.zeros((1, hidden_dim))
W2 = np.random.randn(hidden_dim, output_dim) * 0.1
b2 = np.zeros((1, output_dim))

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def forward(X):
    # 前向传播
    z1 = X @ W1 + b1
    a1 = np.maximum(0, z1)  # ReLU
    z2 = a1 @ W2 + b2
    a2 = sigmoid(z2)        # 输出概率
    cache = (X, z1, a1, z2, a2)
    return a2, cache

def loss_fn(y_pred, y_true):
    # 二分类交叉熵
    eps = 1e-8
    return -np.mean(y_true*np.log(y_pred+eps) + (1-y_true)*np.log(1-y_pred+eps))

def backward(cache, y_pred, y_true):
    # 反向传播，计算梯度
    X, z1, a1, z2, a2 = cache
    m = X.shape[0]
    dz2 = (a2 - y_true) / m                   # dL/dz2
    dW2 = a1.T @ dz2
    db2 = np.sum(dz2, axis=0, keepdims=True)
    da1 = dz2 @ W2.T
    dz1 = da1 * (z1 > 0)                      # ReLU 梯度
    dW1 = X.T @ dz1
    db1 = np.sum(dz1, axis=0, keepdims=True)
    return dW1, db1, dW2, db2

def accuracy(y_pred, y_true):
    y_hat = (y_pred >= 0.5).astype(np.int32)
    return (y_hat == y_true).mean()

for epoch in range(1, epochs+1):
    y_pred, cache = forward(X)
    loss = loss_fn(y_pred, y)
    acc = accuracy(y_pred, y)

    dW1, db1, dW2, db2 = backward(cache, y_pred, y)
    # 参数更新（梯度下降）
    W1 -= lr * dW1
    b1 -= lr * db1
    W2 -= lr * dW2
    b2 -= lr * db2

    if epoch % 300 == 0:
        print(f"Epoch {epoch:4d} | Loss: {loss:.4f} | Acc: {acc:.3f}")

print("训练完成。最终准确率：", accuracy(forward(X)[0], y))