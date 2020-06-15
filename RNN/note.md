# RNN

> Time will explain.

​		RNN是一种语言模型。接下来，我们先探讨什么是语言模型，继而探讨RNN的各种结构，它是如何实现不同任务的，它的缺陷以及改进。以及pytroch的代码实现。

## 1. N-gram Language Models

> Predicting is difficult——especially about the future.	

​		尽管预测是很困难的，但是有些任务却看上去比较简单，比如预测下一个词，

​													*上课了，同学们都翻开___*

​		这里，我们或许都会认为是空白处应该是"书"，而不是其它东西。为什么，<font color = "red">因为"书"的可能性最大，也就是概率最大。</font>而一个为一个序列分配概率的模型，我们就称之为**语言模型**。

### 	1.1 N-grams

​		N-gram语言模型就是最简单的一种语言模型。这里中的N指的就是要分配概率的序列长度，为什么要使用N-grams模型得？答案是因为**approximate**。

​		尝试考虑一下，如果我们想知道：在句子"*its water is so transparent that__"，空格处为"*the*"的概率有多大？那么，我们首先定义一个概率用于表示这样的任务：
$$
P(w|h),  \ \text{表示给定序列h的条件下，词w出现的概率}
$$
​		那么，一种计算方法就是，
$$
P(\text{the}|\text{its water is so transparent that})
=\frac{\text{num(its water is so transparent that the)}}{\text{num(its water is so transparent that)}}
$$
​		也就是，句子"*its water is so transparent that the*"出现的总数除以"*its water is so transparent that the*"，想想都觉得可怕，计算难度太大了。

​		相似地，如果我想知道一个完整序列的概率，类似于$p(\text{its water is so transparent that the})$。看上去，概率的链式法则貌似能帮上忙。
$$
P(X_1,X_2,\cdots,X_n)=P(X_1)P(X_2|X_1)P(X_3|(X_2,X_1))\cdots P(X_n|(X_1,\cdots,X_{n-1}))
$$
​		但是，实际上，对于概率$P(X_n|(X_1,\cdots,X_{n-1}))$又回到了上面问题。还是难以解决。

​		那么，N-grams模型就派上用场了。<font color = "red">首先，N-grams模型认为，计算给定序列h的条件下，词w出现的概率并不用完整的序列h，仅仅只需要预测词的上一个词就行了。</font>也就是：
$$
P(w|h)\approx p(w_n|w_{n-1})
$$
​		这样，
$$
P(\text{the}|\text{its water is so transparent that}) \approx p(\text{the}|\text{that})
$$
​		这里，我们使用了N=2的2-gram(bigram)模型预测下一个词的条件概率。当然，你也可以使用N=3的3-gram(trigram)模型去预测这个条件概率。
$$
P(w_n|(w_{n-1},w_{n-2}))
$$
​		那么，如果我们使用bigram去估计完整序列，那么
$$
\begin{equation}
\begin{aligned}
&P(X_1,X_2,\cdots,X_n)=P(X_1)P(X_2|X_1)P(X_3|(X_2,X_1))\cdots P(X_n|(X_1,\cdots,X_{n-1})) \\
&\approx P(X_1)P(X_2|X_1)P(X_3|X_2)...P(X_n|X_{n-1})\\
&\approx \prod_{k=1}^{n}P(w_k|w_{k-1})
\end{aligned}
\end{equation}
$$
​		这样计算就简单多了。在实际计算中，我们需要为每个序列的前面添加一个符号<s>，末尾添加一个符号<e>。

## 2. RNN: Recurrent Neural Net

​		语言本身就具有时间上的先后顺序。当我们理解语言或者说话时，我们都在处理着未知长度的连续输入。就像你也不知道我这句话会讲多久。(反手就是流畅的16个字符)尽管，你确实可以从任意地方看这篇文章，不过大部分情况下都是从头开始(应该不会有人从末尾开始吧？不会吧？不会吧？)。

###  2.1 Elman Networks

​		首先，我们还是从最简单的RNN的结构——Elman Networks开始讲起。

<img src="C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20200614223837906.png" alt="image-20200614223837906" style="zoom:43%;" />

<center>图(1)</center>

<img src="C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20200614224002564.png" alt="image-20200614224002564" style="zoom:50%;" />

<center>图(2)</center>

​		这里，图(1)是未展开图，而图(2)是展开图。我们就以Elman来讲下前向训练的过程。而反向传播我们在后面再进行展开，我们单独再展开。可能到这里大家还看不出RNN与之前的语言模型的联系，接下来会逐一讲解。	

​		首先，我们定义一些变量。

* 输入序列$x$，$x^{t}$表示序列x中的第t个词。

* $h$为隐层。

* $U$为输入到隐层的映射矩阵。

* $W$为隐层到隐层的矩阵。

* $V$为隐层到输出层的映射矩阵。

  ​	注意，这里的矩阵$W$，$U$和$V$在前向传播过程中是相同的。所以：
  $$
  \begin{equation}
  \begin{aligned}
  a^{t} &= Wh^{(t-1)}+Ux^{(t)} \\
  h^{t} &= g(a^{(t)}), \ \ \text{eg.} \ \ h^{t} = \text{tanh}(a^{t}) \\
  o^{t} &=Vh \\
  \hat{y} &=\text{softmax}(o^{t})\\
  L^{(t)} &=-\log{\hat{y}}
  
  \end{aligned}
  \end{equation}
  $$
  ​		损失的计算为：
  $$
  L(\{x^{(1)},\cdots,x^{(n)}\},\{y\}) = \sum_{t}L^{(t)}
  $$
  ​		所以，其实在训练过程中，我们只需要进行迭代处理就可以了，伪代码为：

  ```python
  def forward():
      h = [] 
  	h0 = [0] * hidden_size # 一般情况下，我们初始化隐层的值为0，隐层的长度由训练时定义
      h.append(h0)
      for i in range(len(x)):
          h[1] <- tanh(U*h[i - 1] + Wx[i])
          o[t] <- V*h[t]
          y_hat <- softmax(o[t])
  ```

   		从上面的伪代码中，我们也可以看到，训练Elman的过程中，只能时顺序进行的，并不能并行运算，这样将大大增加训练的时间。接下来介绍一种**teacher forcing**训练的RNN的结构，它的结构提供了并行训练的可能。

### 2.2 Tearcher Forcing and Netwoks with Output Recurrent

  ​		另外一种RNN的结构是将输出层$o$做为隐层的一个输入。如下图所示：

  ​												<img src="C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20200615091156452.png" alt="image-20200615091156452" style="zoom:43%;" />

  <center>图(3)</center>

  ​		因为缺少了hidden-to-hidden，而是将上一个输出层作为隐层的一个输入，那么久需要输出层能够捕获这样之前的输入信息，因为RNN很重要的一个理念就是用之前的输入信息来预测未来。但是，输出层只是训练来匹配训练集的，因此输出层$o$其实携带很少之前的信息。<font color = "red">除非输出层o的维度很高并且携带信息丰富，否则会丢失之前的信息。</font>

  ​		像这种将输出返回到隐层的RNN模型，是用**Teacher Forcing + BPTT**的方法去训练和求导的。

  											<img src="C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20200615085437349.png" alt="image-20200615085437349" style="zoom:50%;" />

  <center>图(4)</center>

  ​		从图(4)我们可以看到，在训练期间，我们是将正确的标签$y^{(t-1)}$作为下一个隐状态$h^{(t)}$的一个输入。我们知道，$y^{(t-1)}$是事先已经知道的，所以隐状态$h$可以一次性的全部计算出来，就实现了并行训练。尽管在测试上需要顺序进行，但时间其实都是耗费在训练期间的。在使用反向传播求梯度时，我们还是要使用**BPTT**来求解参数$U,V,W$的。

## 3. Applications of Recurrent Neural Networks

  ​		RNN已经被证明在很多任务——诸如，序列标注、文本分类等——非常有效。其它Sequence-to-Sequence模型的应用等讲到再说。

### 3.1 RNNs for Sequence Classification

  ​		我们先来看一个比较简单的例子——用于分类任务时，RNNs时如何工作的。

  					<img src="C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20200615100221150.png" alt="image-20200615100221150" style="zoom:50%;" />

  <center>图(5)</center>

  ​		我们可以看到，这个这个RNN时是没有中间输出，我们仅利用最后的隐层输出$h_n$作为分类器的输入，因此用于训练网络中权重的损失函数完全基于最终的文本分类任务。来自分类器的误差信号被反向传播到前馈分类器的权值，到它的输入，然后到RNN的三组权值。这种使用downstream application(下游应用程序)的损失来通过网络一直调整权重的训练方法被称为**端到端训练(end-to-end)**。

​		接下来，我们就来直接看例子。

  ​		

### 3.2 **Sequence Labeling**

  ​		序列标注任务里，我们等到谈双向RNN时再一并聊起。

  

## 4. Deep Networks: Stacked and Bidirectation RNNs

​		就像图(5)看到的，RNNs能和多种网络进行组合，形成更加复杂的模型。这里介绍使用RNNs来处理自然语言的两种更加常见的网络框架。

### 4.1 **Stacked RNNs**

​		如图(2)所示，我们的长度为N的x序列就有长度为N的隐层h。隐层的信息非常丰富，代表了输入到目前为止的信息。那么，如果我们把这个隐层做为另外一层RNNs的输入，如图(6)所示，那么就能表征更深层次的信息。

<img src="C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20200615110112822.png" alt="image-20200615110112822" style="zoom:50%;" />

​		已经有人证明，在很多任务里，stacked RNNs的性能时优越于single-layer RNNs的。成功的原因之一就是<font color="red">更深的网络能够去表征不同层级之间的信息。</font>但是，随着stack层数的增加，训练带来的耗费也迅速增加。

### 4.2 Bidirectaional RNNs

​		