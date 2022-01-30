## USTC 2021秋《运筹学》课程作业及小测题目

### Ch1 线性规划

**E1.1**	证明凸集表示定理 ：设![](http://latex.codecogs.com/svg.latex?\mathcal{S}=\{\textbf x|A\textbf x=\mathbf b,\textbf x\ge0\})为非空多面集，则有 

- 极点集非空且有限 

- 极方向集合为空集的充要条件是![](http://latex.codecogs.com/svg.latex?\mathcal{S})有界  

- 若![](http://latex.codecogs.com/svg.latex?\mathcal{S})无界，则存在有限个极方向  

- ![](http://latex.codecogs.com/svg.latex?X\in\mathcal{S})的充要条件是 

  ![](http://latex.codecogs.com/svg.latex?\mathbf x=\sum_{i=1}^{k} \lambda_{i} \mathbf x^{(i)}+\sum_{j=1}^{\ell} \mu_{j} \mathbf d^{(j)})

  其中![](http://latex.codecogs.com/svg.latex?\lambda_{i} \geq 0, i=1, \ldots, k, \sum_{i=1}^{k} \lambda_{i}=1, \mu_{j} \geq 0, j=1, \ldots, \ell)。

**E1.2**	给出

![](http://latex.codecogs.com/svg.latex?\begin{array}{c}
\min -x_{1}+3 x_{2} \\
\text { s.t. } x_{1}+x_{2} \leq 8 \\
x_{2} \leq 2 \\
x_{1}, x_{2} \geq 0
\end{array})

的所有极点和可行基解。

**E1.3**	证明线性规划模型的基本可行解集与其可行域的极点集是等价的。

**E1.4**	证明单纯形法转轴运算得到的

![](http://latex.codecogs.com/svg.latex?x=(x_{B_1},\cdots,x_{B_{r-1}},0,x_{B_{r+1}},\cdots,x_{B_{m}},0,\cdots,x_p,\cdots,0)^T)

是一个新的可行基解。

**E1.5**	先列出单纯形法的计算步骤，然后求解线性规划问题：

![](http://latex.codecogs.com/svg.latex?\begin{array}{c}
\min -4 x_{1}-x_{2} \\
\text { s.t. }-x_{1}+2 x_{2} \leq 4 \\
2 x_{1}+3 x_{2} \\
x_{1}-x_{2} \leq 3 \\
x_{1}, x_{2} \geq 0
\end{array})

**E1.6**	证明在执行主元消去法前后两个不同可行基下的判别系数和目标函数值有如下关系 

![](http://latex.codecogs.com/svg.latex?\begin{aligned}
\left(z_{j}-c_{j}\right)^{\text {new }} &=\left(z_{j}-c_{j}\right)-\frac{y_{r j}}{y_{r p}}\left(z_{p}-c_{p}\right), \\
\left(c_{B}^{T} B^{-1} b\right)^{\text {new }} &=c_{B}^{T} B^{-1} b-\frac{\bar{b}_{r}}{y_{r p}}\left(z_{p}-c_{p}\right).
\end{aligned})

**E1.7**	证明线性规划问题的对偶定理，即原问题(LP)和对偶问题(DP)只要一方有最优解，则另一方也有最优解且此时两方的最优值一致。

---------

### Ch2 网络最优化

**E2-1**	依据对偶理论，给出最短路径问题的一种求解方法。  

**E2-2**	证明最大流最小割定理：任意一个流网络的最大流量等于该网络的最小割的
容量。

**E2-3**	验证下式给出的![](http://latex.codecogs.com/svg.latex?x^{\prime})是原网络![](http://latex.codecogs.com/svg.latex?\mathcal{N})的可行流，并且其流值为![](http://latex.codecogs.com/svg.latex?x^{\prime}(s)=x(s)+\Delta)。

![](http://latex.codecogs.com/svg.latex?x^{\prime}(e)=\left\{\begin{array}{ll}
x(e)+\Delta, & (v, w) \in \pi\ (\text { Rule }1) \\
x(e)-\Delta, & (w, v) \in \pi\ (\text { Rule }2) \\
x(e), & \text {other}
\end{array}\right.)

**E2-4**	通过构造说明最小成本流问题作为其特殊情况包含：最短路问题和最大流问题。  

**E2-5**	**【课堂小测】**证明最小成本循环流问题与最小成本流问题具有等价的模型化能力。  

--------

### Ch3 动态规划

**E3-1**	设现有一台2年龄的设备，另规定5年龄的设备必须更换。在规划期购置新设备的成本分别是  

![](http://latex.codecogs.com/svg.latex?(p_1,p_2,p_3,p_4,p_5)=(100,105,110,115,120))

试构建如下设备更新的动态规划模型并求其最优更新策略。  

| 设备年龄 | 残值 | 运行费用 |
| :------: | :--: | :------: |
|    0     |  -   |    30    |
|    1     |  50  |    40    |
|    2     |  25  |    50    |
|    3     |  10  |    75    |
|    4     |  5   |    90    |
|    5     |  2   |    -     |

----

### Ch5 无约束最优化

**E5.1**	写出基于Wolfe-Powell准则的非精确一维搜索算法中插值多项式![](http://latex.codecogs.com/svg.latex?p^{(1)}(t),p^{(2)}(t))的具体表达式。  

**E5.2**	证明基于Goldstein准则的非精确一维搜索算法的全局收敛性。  

**E5.3**	试将非线性方程组求根![](http://latex.codecogs.com/svg.latex?F(\bf x)=0)的牛顿迭代，用于求解无约束最优化问题![](http://latex.codecogs.com/svg.latex?\min_{\bf x\in\mathbb{R}^{n}}f(\bf x))。给出相应的迭代格式并说明理由。

**E5.4**	证明对称秩一牛顿法具有遗传性和二次终止性  

**E5.5**	**【课堂小测】**利用秩一校正的求逆公式（Sherman-Morrison定理），由![](http://latex.codecogs.com/svg.latex?H_{k+1}^{DFP})推导![](http://latex.codecogs.com/svg.latex?B_{k+1}^{DFP})。

**E5.6**	证明共轭梯度法的性质定理：

设目标函数![](http://latex.codecogs.com/svg.latex?f(\textbf  x)=\frac{1}{2}\textbf x^TG\textbf x+\textbf c^T\textbf x)，则采用精确一维搜索的共轭梯度法经![](http://latex.codecogs.com/svg.latex?m\le n)步迭代后终止，且对所有的![](http://latex.codecogs.com/svg.latex?1\le k\le m)成立下列关系式：

![](http://latex.codecogs.com/svg.latex?\begin{array}{l}
\textbf d^{(k)^{\top}} G \textbf d^{(j)}=0, j=0,1, \ldots, k-1 \\
\textbf g^{(k)^{\top}} \textbf g^{(j)}=0, j=0,1, \ldots, k-1 \\
\textbf d^{(k)^\top \top} \textbf g^{(k)}=-\textbf g^{(k)^{\top}} \textbf g^{(k)} \\
\operatorname{span}\left\{\textbf g^{(0)}, \textbf g^{(1)}, \ldots, \textbf g^{(k)}\right\}=\operatorname{span}\left\{\textbf g^{(0)}, G \textbf g^{(0)}, \ldots, G^{k} \textbf g^{(0)}\right\} \\
\operatorname{span}\left\{\textbf d^{(0)}, \textbf d^{(1)}, \ldots, \textbf d^{(k)}\right\}=\operatorname{span}\left\{\textbf d^{(0)}, G \textbf g^{(0)}, \ldots, G^{k} \textbf g^{(0)}\right\}
\end{array})

**E5.7**	**【课堂小测】**在信赖域方法中，请给出一种与调整信赖域半径等效的自适应模式算法。

**E5.8**	证明折线法（信赖域方法）子问题模型的函数单调性。 

**E5.9**	**【课堂小测】**请给出![](http://latex.codecogs.com/svg.latex?H_{k+1}^{BFGS})的对称秩二校正的特解，即![](http://latex.codecogs.com/svg.latex?a,\mathbf u,b,\mathbf v)。

------

### Ch6 二次规划

**E6.1**	证明积极集基本定理：设![](http://latex.codecogs.com/svg.latex?\mathbf x^*)是一般的二次规划问题的局部极小点，则![](http://latex.codecogs.com/svg.latex?\mathbf x^*)也必是等式约束问题

![](http://latex.codecogs.com/svg.latex?\begin{array}{ll}
\min & Q(\mathbf x)=\frac{1}{2} \mathbf x^{\top} G \mathbf x+\mathbf c^{\top} \mathbf x \\
\text { s.t. } & \mathbf a_{i}^{\top} \mathbf x=b_{i},  i \in \mathcal{E} \cup \mathcal{I}\left(\mathbf x^{*}\right)
\end{array})

的局部极小点。反之，如果![](http://latex.codecogs.com/svg.latex?\mathbf x^*)是一般问题的可行点，同时是上述问题的K-T点，且相应的Lagrange乘子![](http://latex.codecogs.com/svg.latex?\lambda^*)满足![](http://latex.codecogs.com/svg.latex?\lambda_i^*\ge0,i\in\mathcal{I}(\mathbf x^*))，则![](http://latex.codecogs.com/svg.latex?\mathbf x^*)必是原问题的K-T点。

一般的二次规划为 

![](http://latex.codecogs.com/svg.latex?\begin{aligned}
\min\quad Q(\mathbf x) &=\frac{1}{2} \mathbf x^{\top} G \mathbf x+\mathbf c^{\top} \mathbf x \\
\text { s.t. }\quad \mathbf a_{i}^{\top} \mathbf x &=b_{i}, i \in \mathcal{E}=\left\{1, \ldots, m_{e}\right\} \\
\mathbf a_{i}^{\top} \mathbf x & \geqslant b_{i}, i \in \mathcal{I}=\left\{m_{e}+1, \ldots, m\right\}
\end{aligned}  )

**E6.2**	**【课堂小测】**考虑等式约束问题 

![](http://latex.codecogs.com/svg.latex?\begin{array}{ll}
\min & \frac{1}{2} \mathbf s^{T} G \mathbf s+(G \mathbf x^{k} +\mathbf c)^{T} \mathbf s\\
\text { s.t.}& \mathbf a_{i}^{T} \mathbf s=0, i \in \mathcal{E}_{k}
\end{array})

求得其解为![](http://latex.codecogs.com/svg.latex?\mathbf s^k)，及其相应的Lagrange乘子![](http://latex.codecogs.com/svg.latex?\lambda_i^k)，![](http://latex.codecogs.com/svg.latex?i \in \mathcal{E}_{k})。

若![](http://latex.codecogs.com/svg.latex?\mathbf s^k=0)，且![](http://latex.codecogs.com/svg.latex?\lambda_i^k\ge0)，![](http://latex.codecogs.com/svg.latex?i \in \mathcal{E}_{k})不成立，则由，![](http://latex.codecogs.com/svg.latex?\lambda_{i_{q}}^{k}=\min _{i \in \mathcal{I}\left(x^{k}\right)} \lambda_{i}^{k}<0)确定![](http://latex.codecogs.com/svg.latex?i_q)，那么如下问题 

![](http://latex.codecogs.com/svg.latex?\begin{array}{ll}
\min & \frac{1}{2} \mathbf s^{T} G \mathbf s+(G \mathbf x^{k} +\mathbf c)^{T} \mathbf s\\
\text { s.t.} & \mathbf a_{i}^{T} \mathbf s=0, i \in \hat{\mathcal{E}}=\mathcal{E}_{k} \backslash\{i_{q}\}
\end{array}) 的解![](http://latex.codecogs.com/svg.latex?\mathbf{\hat s})是原问题在当前点![](http://latex.codecogs.com/svg.latex?\mathbf{x}^k)处的可行方向，即![](http://latex.codecogs.com/svg.latex?\mathbf a_{i_q}^{T}\mathbf{\hat s}\ge0)。

---------

### Ch7 非线性约束最优化

**E7.1**	证明![](http://latex.codecogs.com/svg.latex?\psi(\textbf x,\lambda)=\parallel\bigtriangledown f(\textbf x)-A(\textbf x)^T\lambda\parallel^2+ \parallel\bf c(x)\parallel^2)中定义的![](http://latex.codecogs.com/svg.latex?\psi(\textbf x,\lambda))是关于Lagrange-Newton法的下降函数。  

**E7.2**	证明罚函数法求解带误差界近似问题的算法有限终止性。 

**E7.3**	给出约束最优化问题的二阶充分最优性条件，并用于说明增广Lagrange函数的极小点与原问题最优解的等价性。

**E7.4**	**【课堂小测】**设![](http://latex.codecogs.com/svg.latex?\sigma_{k+1}>\sigma_{k}>0)，则有

![](http://latex.codecogs.com/svg.latex?\begin{aligned}
P_{\sigma_{k}}[\mathbf{x}(\sigma_{k})] &\leq P_{\sigma_{k+1}}[\mathbf{x}(\sigma_{k+1})], \\
\parallel\mathbf{c}[\mathbf{x}(\sigma_{k})]_{-}\parallel &\geq \parallel\mathbf{c}[\mathbf{x}(\sigma_{k+1})]_{-}\parallel, \\
f[\mathbf{x}(\sigma_{k})] &\leq f[\mathbf{x}(\sigma_{k+1})].
\end{aligned})

**E7.5**	令![](http://latex.codecogs.com/svg.latex?\delta=\left\|c[x(\sigma)]_{-}\right\|)，则![](http://latex.codecogs.com/svg.latex?\bf x(\sigma))也是约束问题

![](http://latex.codecogs.com/svg.latex?\begin{array}{ll}
\min \quad f(x) \\
\text { s.t.}\quad\left\|c(x)_{-}\right\| \leqslant \delta 
\end{array})

的最优解。

**E7.6**	设![](http://latex.codecogs.com/svg.latex?\theta_{k}>\theta_{k+1}>0)，记![](http://latex.codecogs.com/svg.latex?\textbf x(\theta)=\arg \min_{\textbf x}B(\textbf x,\theta))则有

![](http://latex.codecogs.com/svg.latex?\begin{aligned}
B[\mathbf{x}(\theta_{k}), \theta_{k}] & \geq B[\mathbf{x}(\theta_{k+1}), \theta_{k+1}], \\
\psi[\mathbf{x}(\theta_{k})] & \leq \psi[\mathbf{x}(\theta_{k+1})], \\
f[\mathbf{x}(\theta_{k})] & \geq f[\mathbf{x}(\theta_{k+1})] .
\end{aligned})

**E7.7**	**【课堂小测】**设![](http://latex.codecogs.com/svg.latex?\mathbf x^*)和![](http://latex.codecogs.com/svg.latex?\lambda^*)满足如下等式约束问题局部最优解的二阶充分条件，则存在![](http://latex.codecogs.com/svg.latex?\bar \sigma)使得当![](http://latex.codecogs.com/svg.latex?\sigma>\bar \sigma)时，![](http://latex.codecogs.com/svg.latex?\mathbf x^*)是函数![](http://latex.codecogs.com/svg.latex?P(\mathbf x,\lambda^*,\sigma))的严格局部极小点。

