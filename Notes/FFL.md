这篇论文标题为“Transformer Feed-Forward Layers Are Key-Value Memories”（Transformers前馈层是键值存储），研究了前馈层在Transformer语言模型中的作用，提出它们实际上充当了键值存储的功能。以下是详细解释：

### 论文摘要
- 前馈层占Transformer模型参数的三分之二，但其作用尚未深入研究。
- 本文表明，在基于Transformer的语言模型中，前馈层作为键值存储操作，每个键与训练示例中的文本模式相关，每个值对输出词汇表分布产生影响。
- 实验显示，学习到的模式对人类是可解释的，较低层次捕获浅层模式，较高层次学习到更多语义模式。
- 这些值通过引导在上层模型中出现的每个模式后的概率分布来补充键的输入模式。
- 前馈层的输出是其记忆的组合，通过残差连接在模型的各层中逐步精炼，最终形成最终的输出分布。

### 主要内容
1. **引言**：
   - Transformer模型在自然语言处理领域取得了巨大成功，尤其是自注意力机制的引入。
   - 大量文献分析了自注意力层的功能，但前馈层的作用却没有得到充分探索。

2. **前馈层作为非标准化的键值存储**：
   - 前馈层是逐位置的函数，每个输入向量独立处理。
   - 前馈层可以表示为：FF(x) = f(x · K>) · V，其中K和V是参数矩阵，f是非线性函数。
   - 前馈层与键值存储非常相似，唯一的不同在于前馈层没有使用softmax作为非线性函数。

3. **键捕获输入模式**：
   - 假设键向量K在输入序列上充当模式检测器，每个键向量对应特定的输入前缀模式。
   - 通过分析训练好的语言模型的前馈层中的键，可以看到每个键都与一组人类可解释的输入模式相关联。

4. **值表示分布**：
   - 每个值可以看作是对输出词汇表的概率分布，特别是在模型的上层，这种分布与对应键的模式的下一个词的分布有关。
   - 实验证明，在上层模型中，值的分布确实与键的模式对应。

5. **记忆的聚合**：
   - 前馈层的输出是多个记忆的加权和，然后通过残差连接逐层精炼，形成模型的最终输出。
   - 研究显示，在每一层中，前馈层输出的预测通常与单个记忆的预测不同，而是多个记忆的组合。

### 结论
- 前馈层在Transformer语言模型中充当了模式检测器的角色，模型的最终输出分布是从下到上逐步构建的。
- 这些发现为未来的研究开辟了重要的方向，包括层嵌入空间的研究、超越语言建模的应用以及在实际应用中的影响。

通过这些分析，论文深入揭示了前馈层在Transformer模型中的作用，为理解和改进这些模型提供了新的视角。

---

前馈层与键值存储的相似之处在于它们都使用键（keys）和值（values）对输入进行处理，但它们的主要区别在于前馈层没有使用softmax作为非线性函数。以下是详细的解释和举例：

### 键值存储的工作原理
1. **键值对**：
   - 键值存储系统由多个键值对（key-value pairs）组成，每个键和值都是向量。
   - 键向量 \( k_i \) 和值向量 \( v_i \) 组成一个键值对。

2. **输入处理**：
   - 给定一个输入向量 \( x \)，计算该输入与每个键的相似度，通常通过点积计算：
     \[
     p(k_i | x) \propto \exp(x \cdot k_i)
     \]
   - 使用softmax函数将这些相似度转换为概率分布：
     \[
     p(k_i | x) = \frac{\exp(x \cdot k_i)}{\sum_j \exp(x \cdot k_j)}
     \]
   - 计算输出为所有值向量的加权和：
     \[
     MN(x) = \sum_i p(k_i | x) v_i
     \]

### 前馈层的工作原理
1. **前馈层结构**：
   - 前馈层包含两个参数矩阵：键矩阵 \( K \) 和值矩阵 \( V \)。
   - 键矩阵 \( K \) 包含多个键向量，每个向量表示一个键；值矩阵 \( V \) 包含多个值向量，每个向量表示一个值。

2. **输入处理**：
   - 给定一个输入向量 \( x \)，计算该输入与键矩阵的乘积，得到一个中间向量 \( m \)：
     \[
     m = f(x \cdot K^T)
     \]
     这里的 \( f \) 通常是ReLU函数，即 \( f(z) = \max(0, z) \)。
   - 计算输出为中间向量 \( m \) 和值矩阵 \( V \) 的乘积：
     \[
     FF(x) = m \cdot V
     \]

### 具体例子解释
假设我们有一个简单的键值存储和一个前馈层，分别由两个键值对和两个参数矩阵组成：

#### 键值存储
- 键向量： \( k_1 = [1, 0], k_2 = [0, 1] \)
- 值向量： \( v_1 = [2, 3], v_2 = [4, 1] \)
- 输入向量： \( x = [0.5, 0.5] \)

1. 计算输入与键的相似度：
   \[
   p(k_1 | x) = \frac{\exp(0.5 \cdot 1 + 0.5 \cdot 0)}{\exp(0.5) + \exp(0.5)} = \frac{\exp(0.5)}{2 \cdot \exp(0.5)} = 0.5
   \]
   \[
   p(k_2 | x) = \frac{\exp(0.5 \cdot 0 + 0.5 \cdot 1)}{\exp(0.5) + \exp(0.5)} = \frac{\exp(0.5)}{2 \cdot \exp(0.5)} = 0.5
   \]

2. 计算输出为值向量的加权和：
   \[
   MN(x) = 0.5 \cdot v_1 + 0.5 \cdot v_2 = 0.5 \cdot [2, 3] + 0.5 \cdot [4, 1] = [3, 2]
   \]

#### 前馈层
- 键矩阵： \( K = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} \)
- 值矩阵： \( V = \begin{bmatrix} 2 & 3 \\ 4 & 1 \end{bmatrix} \)
- 输入向量： \( x = [0.5, 0.5] \)

1. 计算输入与键矩阵的乘积并应用ReLU：
   \[
   m = f(x \cdot K^T) = f([0.5, 0.5] \cdot \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}) = f([0.5, 0.5]) = [0.5, 0.5]
   \]

2. 计算输出为中间向量 \( m \) 和值矩阵 \( V \) 的乘积：
   \[
   FF(x) = m \cdot V = [0.5, 0.5] \cdot \begin{bmatrix} 2 & 3 \\ 4 & 1 \end{bmatrix} = [3, 2]
   \]

### 对比与总结
- 在键值存储中，输入与键的相似度通过softmax归一化为概率分布，进而加权求和得到输出。
- 在前馈层中，输入与键的相似度通过ReLU非线性函数处理，得到中间向量，然后与值矩阵相乘得到输出。
- 两者的主要区别在于前馈层没有使用softmax进行归一化处理，而是直接应用ReLU函数，这使得前馈层的输出在数值上没有归一化，但在功能上实现了类似于键值存储的效果。


---

前馈层确实是全连接层，但在本文中，它们被解释为具有特定功能的全连接层，即键值存储。这种视角帮助我们理解前馈层如何处理信息。以下是详细的解释：

### 前馈层作为全连接层
在Transformer模型中，前馈层（Feed-Forward Layer，FFN）通常由两个全连接层（也称为线性层）组成，中间有一个非线性激活函数（如ReLU）。公式表示如下：

1. 首先，输入向量 \( x \) 通过第一个全连接层，得到中间向量 \( z \)：
   \[
   z = x \cdot W_1 + b_1
   \]
   这里 \( W_1 \) 是第一个全连接层的权重矩阵， \( b_1 \) 是偏置向量。

2. 然后，应用非线性激活函数（如ReLU）：
   \[
   a = \text{ReLU}(z)
   \]

3. 最后，输出通过第二个全连接层，得到最终的前馈层输出：
   \[
   y = a \cdot W_2 + b_2
   \]
   这里 \( W_2 \) 是第二个全连接层的权重矩阵， \( b_2 \) 是偏置向量。

### 前馈层作为键值存储
本文将前馈层解释为键值存储，是为了揭示其在信息处理中的具体功能。具体来说，前馈层可以看作是通过键（keys）和值（values）对输入进行加权和组合，从而生成输出。这种视角强调了前馈层在捕捉和存储输入模式方面的作用。

1. **键和值的表示**：
   - 键矩阵 \( K \) 和值矩阵 \( V \) 分别对应前馈层中的两个全连接层的权重矩阵。
   - 键矩阵 \( K \) 表示为 \( W_1 \)，值矩阵 \( V \) 表示为 \( W_2 \)。

2. **输入处理**：
   - 给定输入向量 \( x \)，首先计算输入与键矩阵的乘积：
     \[
     z = x \cdot K^T = x \cdot W_1^T
     \]

   - 然后，应用非线性激活函数（如ReLU），得到激活向量 \( a \)：
     \[
     a = \text{ReLU}(z)
     \]

   - 最后，激活向量与值矩阵相乘，得到前馈层的输出：
     \[
     y = a \cdot V = \text{ReLU}(x \cdot K^T) \cdot W_2
     \]

### 举例说明
假设有一个简单的前馈层，由以下参数组成：
- 键矩阵（对应 \( W_1 \)）： \( K = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} \)
- 值矩阵（对应 \( W_2 \)）： \( V = \begin{bmatrix} 2 & 3 \\ 4 & 1 \end{bmatrix} \)
- 输入向量： \( x = [0.5, 0.5] \)

1. **计算键矩阵的乘积**：
   \[
   z = x \cdot K^T = [0.5, 0.5] \cdot \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} = [0.5, 0.5]
   \]

2. **应用ReLU激活函数**：
   \[
   a = \text{ReLU}(z) = \text{ReLU}([0.5, 0.5]) = [0.5, 0.5]
   \]

3. **计算值矩阵的乘积**：
   \[
   y = a \cdot V = [0.5, 0.5] \cdot \begin{bmatrix} 2 & 3 \\ 4 & 1 \end{bmatrix} = [0.5 \cdot 2 + 0.5 \cdot 4, 0.5 \cdot 3 + 0.5 \cdot 1] = [3, 2]
   \]

### 总结
- 前馈层确实是全连接层，但在本文中，通过将其视为键值存储，我们能够更好地理解它在信息处理中的角色。
- 键值存储的视角揭示了前馈层如何捕捉输入模式并生成输出，这对于深入理解Transformer模型的工作原理是非常有价值的。

---

键捕获输入模式：

假设键向量K在输入序列上充当模式检测器，每个键向量对应特定的输入前缀模式。
通过分析训练好的语言模型的前馈层中的键，可以看到每个键都与一组人类可解释的输入模式相关联。
值表示分布：

每个值可以看作是对输出词汇表的概率分布，特别是在模型的上层，这种分布与对应键的模式的下一个词的分布有关。
实验证明，在上层模型中，值的分布确实与键的模式对应。

---

为了更好地理解前馈层如何通过引导上层模型中出现的每个模式后的概率分布来补充键的输入模式，并通过残差连接在模型的各层中逐步精炼，最终形成最终的输出分布，我们可以从以下几个方面进行详细解释。

### 前馈层的作用

1. **键-值机制**：
   - **键（Key）**：捕获输入中的特定模式，如n-gram、短语或语义模式。
   - **值（Value）**：对于捕获的模式生成一个输出词汇表的概率分布。

   具体来说，键矩阵 \(K\) 中的每个键向量 \(k_i\) 对应特定的输入模式，值矩阵 \(V\) 中的每个值向量 \(v_i\) 对应输出词汇表的概率分布。

2. **输入处理和输出生成**：
   - 给定输入向量 \(x\)，首先计算其与键矩阵 \(K\) 的相似度，生成中间向量 \(z\)：
     \[
     z = x \cdot K^T
     \]
   - 应用非线性激活函数（如ReLU）得到激活向量 \(a\)：
     \[
     a = \text{ReLU}(z)
     \]
   - 将激活向量 \(a\) 与值矩阵 \(V\) 相乘，得到前馈层的输出 \(y\)：
     \[
     y = a \cdot V
     \]

### 前馈层输出的记忆组合

1. **记忆组合**：
   - 前馈层的输出 \(y\) 是其记忆（键-值对）的组合。这些记忆的组合体现在中间激活向量 \(a\) 的各个分量对值向量的加权和。

   例如，假设有三个键-值对和一个输入向量 \(x\)，我们得到：
   \[
   z = [z_1, z_2, z_3] = x \cdot K^T
   \]
   \[
   a = \text{ReLU}(z) = [a_1, a_2, a_3]
   \]
   \[
   y = a \cdot V = a_1 v_1 + a_2 v_2 + a_3 v_3
   \]

2. **通过残差连接精炼**：
   - 残差连接（Residual Connections）在Transformer中起到信息传递和精炼的作用。
   - 每一层的输出不仅依赖于当前层的计算，还包括前一层的输出。这使得信息可以在多个层次上流动和融合，从而逐步精炼模型的最终输出。

   在第 \(l\) 层，残差连接的计算如下：
   \[
   x^{(l+1)} = \text{LayerNorm}(x^{(l)} + \text{FF}(x^{(l)}))
   \]
   其中 \(x^{(l)}\) 是第 \(l\) 层的输入，\(\text{FF}(x^{(l)})\) 是前馈层的输出，\(\text{LayerNorm}\) 是层归一化。

### 示例解析

假设我们有一个输入序列通过一个多层的Transformer模型：

1. **第一层前馈层**：
   - 输入向量 \(x^{(0)}\) 与键矩阵 \(K^{(0)}\) 进行计算，生成中间向量 \(z^{(0)}\)。
   \[
   z^{(0)} = x^{(0)} \cdot (K^{(0)})^T
   \]
   - 通过ReLU激活函数生成激活向量 \(a^{(0)}\)：
   \[
   a^{(0)} = \text{ReLU}(z^{(0)})
   \]
   - 计算输出向量 \(y^{(0)}\)：
   \[
   y^{(0)} = a^{(0)} \cdot V^{(0)}
   \]

2. **残差连接**：
   - 将第一层的输出 \(y^{(0)}\) 与输入 \(x^{(0)}\) 相加并进行层归一化，得到下一层的输入 \(x^{(1)}\)：
   \[
   x^{(1)} = \text{LayerNorm}(x^{(0)} + y^{(0)})
   \]

3. **重复多层**：
   - 这个过程在每一层中重复，逐层精炼输入和输出。
   - 每一层的键值存储和前馈计算引入新的信息和模式检测，并在残差连接中融合前一层的信息。

4. **最终输出**：
   - 多层前馈层和残差连接的组合使得模型能够捕获复杂的模式和长距离依赖，从而生成精确的输出词汇表分布。

### 总结

- 前馈层的输出是其记忆的组合，即通过键-值对的操作捕获输入模式并生成对应的输出分布。
- 残差连接允许信息在层之间流动，逐步精炼输出，最终形成精确的输出分布。
- 这种机制使得Transformer模型能够高效地处理复杂的语言任务，捕获多层次的模式和依赖关系。