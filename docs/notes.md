



# Quant Method

记$V$为codebook的code数量，$d$为特征数量，那么Quantizer的核心是如何构建形状词表大小为$V$，特征数量为$d$的codebook，我们记quantize的映射为：
$$
x_{\mathrm{quant}}=f(x, \mathbf W), x, x_{\mathrm{quant}}\in \mathbb R^d, \\
j=\mathrm{argmin}_{k=1}^V d(x, \mathbf W_k), f(x,\mathbf W)=\mathbf W_j, \\
d为某种指标。
$$




## 对词表进行分解

### 方案1(naive)

直接构造$\mathbf W\in \mathbb R^{V\times d}$的矩阵，即：
$$
x_{\mathrm{quant}}=f(x, \mathbf W)\in \mathbb R^d.
$$


### 方案2(词表纬度分解)

对词表纬度进行分解，特征维度进行分解或者不分解都可：
$$
V=\prod_j V_j, \mathbf W_j \in \mathbb R^{V_j \times d_j}.
$$
其中$\mathbf W_j$可以进行参数共享等等。

接下来就是对于输入$x$，找到$x_{\mathrm{quant}}$，提供三种方案：

concat型（对特征维度进行分解）：
$$
\sum d_j =d, \\
x_k \in \mathbb R^{d_k},
x=\left[
\begin{matrix}
x_1 \\
\vdots \\
x_k
\end{matrix}
\right]\in \mathbb R^{d}, \\
x_{\mathrm{quant}}=\left[
\begin{matrix}
f(x_1, \mathbf W_1) \\
\vdots \\
f(x_k, \mathbf W_k)
\end{matrix}
\right]\in \mathbb R^{d}.
$$
加法型（对特征维度不进行分解）：
$$
d_j =d,
x_{\mathrm{quant}}=\sum_{j} f(x, \mathbf W_j) .
$$
残差型（对特征维度不进行分解）：
$$
d_j =d,
r_0=x, r_j = r_{j-1}-\bar x_j, \\
\bar x_j=f(r_{j-1}, W_j), \\
x_{\mathrm{quant}}=\sum_{j=1}^{k}\bar x_j .
$$
说明：

- Var/Rqvae是对$\sum_{j=1}^{s}\bar x_j$和$x$算loss；



### 方案3(特征维度分解)

对特征纬度进行分解
$$
d=\prod_j d_j, \mathbf W_j \in \mathbb R^{V_j \times d_j}.
$$
注意此时词表大小为$\prod_j V_j $，所以可以归结为方案2。



## 如何得到codebook

寻找codebook的核心是下式：
$$
j=\mathrm{argmin}_{k=1}^V d(x, \mathbf W_k), f(x,\mathbf W)=\mathbf W_j, \\
d为某种指标。
$$


### 方案1(可学版本)

$\mathbf W$随机初始化，可学。



### 方案2(固定codebook)

考虑一个种情况，假设我们固定$\mathbf W$，使其在空间中均匀。



#### 例1(笛卡尔坐标)

假设空间为$[0, N), N\in \mathbb N^+$，将其划分为$V$的等距区间：$[iN/V, (i+1)N/V),i=0,\ldots, V-1$，取$\mathbf W_j=\frac{(2i+1)N}{2V}$，
$$
f(x,\mathbf W)=\mathbf W_j.
$$
特别的，取$N=V$，此时$\mathbf W_j=i+\frac 1 2$：
$$
f(x,\mathbf W)=\mathrm{round}(x).
$$
即可得到Fsq。

注意该方案只能处理特征1维情形，要处理高维情形，可以对空间$\Pi_{j=1}^k [0, N)$进行划分，等价的，是对词表维度分解的方案。



#### 例2(球坐标)

根据例1的思路，我们可以考虑$n$维，将输入$x$转换为[球坐标系](https://en.wikipedia.org/wiki/Spherical_coordinate_system)，假设$r=1$：
$$
x=[\theta_1,\ldots, \theta_{d-1}],
\theta_k \in [0, 2\pi].
$$
我们可以在$[0, 2\pi]$区间上进行均匀划分，然后得到量化结果。



## 如何计算argmin





## Loss





