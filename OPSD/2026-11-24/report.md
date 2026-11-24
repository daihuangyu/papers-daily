# OPD & OPSD 精读 · 2026-11-24

## 先读这五篇

### 1. CoRD：让学生走到平时很少尝试的解题路径

学生模型用自己生成的解题过程练习。它如果总是从同一种思路起步，老师就很少有机会在其他解法的中间步骤上指导它。

CoRD 在生成每一步时先让学生提出下一个词，同时根据老师和学生的预测定一个折中的采样目标。只要能满足这个目标，就尽量保留学生的选择。需要替换时，从目标更偏好、学生又选得偏少的词里抽一个。这样能以尽量少的替换让学生走到新路径，再用原来的蒸馏方式学习。[读第 1 篇精读](#paper-cord)

### 2. DWE：减少反向计算，仍保留完整的学习信号

老师会给词表里每个词打概率。学生从这些概率里学习时，要为大量词计算参数该怎么调整，输出层的反向计算因此很贵。

DWE 先照常计算完整词表的学习信号，再把需要往相反方向调整的词分成两边。它按输出层参数的相似程度给词排序，用同一个随机位置从两边各取一个词，并把抽样分散到不同区间。用抽中的词估算反向更新，多次抽样取平均会趋近完整计算的结果。它主要省的是输出层这一步。[读第 2 篇精读](#paper-dwe)

### 3. LCS 参考选择：从成功案例里挑一条能接上学生进度的

同一个工具任务可能有多条成功路线。学生已经打开一个柜子，参考却从另一个柜子取东西。老师照着这条参考纠错，可能会让学生推翻前面已经做对的动作。

作者先收集每道任务的成功记录。学生失败时，就找一条与它共享有序动作比例最高的成功记录。中间允许跳过多余动作，所以绕了一小段弯之后接回原路也能匹配。只有老师在批改时看参考。学生成功时跳过这次纠错，找不到成功记录时就照常让老师评分。[读第 3 篇精读](#paper-lcs)

### 4. Targeted Guidance：在失败的那一轮告诉模型怎么改

如果一个写代码的助手几乎从未用对新工具格式，单靠奖励很难让它运气好到碰见一次正确做法。给整段对话加指导，又可能干扰它已经掌握的操作。

给每项要求准备有效的提示，只加到没达标的那一轮前。让同一个模型读着提示当老师，逐词指导学生原来的回答。同时保留任务和辅助要求的奖励。作者发现，指导过强容易失稳，给失败回合的内部推理也加上指导，能减少风格表现的反复。[读第 4 篇精读](#paper-guidance)

### 5. 教师后门风险：异常可能藏在老师给出的概率里

就算学生生成的回答看起来正常，老师也可能在某种输入条件下偷偷抬高一类回答的概率。学生学习这些概率时，可能连同老师的异常行为一起学走。

作者先写一条意思相同的对照输入。学生照常用原输入作答，老师分别看两种输入，在每个位置预测下一词。两边的概率差超过门槛时，就用对照输入下的预测来训练该位置，其余位置照常训练。对照改写需要保住原意。[读第 5 篇精读](#paper-teacher-risk)

## 本期选题与阅读说明

本期主题：训练走到哪里、反传哪些坐标、选择哪条参考，以及如何控制自蒸馏和教师监督的副作用。

检索窗口为 **2026-10-24 至 2026-11-24**，按 arXiv 首次提交日期核验；本期实际收录 **5 篇新论文**。全部固定阅读 v1，日期不以博客发布时间替代。Targeted Guidance 已有 v2，本期仍明确分析 v1，不把版本更新再算一篇。与 sources.json 和所有方向历史 manifest 按去版本 arXiv ID、DOI、规范 URL、规范标题及别名查重。这里的“新”表示窗口内且本仓库首次收录，并非都在今天首发。

| 论文与原文 | 首投日期（UTC） | 本期要解决的具体问题 |
| --- | --- | --- |
| [CoRD](https://arxiv.org/abs/2611.07959) | 2026-11-05 | 学生走不到的教师路径，怎样获得监督？ |
| [Distillation with Errors（DWE）](https://arxiv.org/abs/2611.27131) | 2026-11-20 | 全词表反传太贵，怎样做无偏稀疏估计？ |
| [Which Successes Make Good References?](https://arxiv.org/abs/2611.13135) | 2026-11-09 | 同一道任务的成功轨迹，哪条适合纠正当前失败？ |
| [Stabilizing On-Policy Targeted Guidance](https://arxiv.org/abs/2611.05266) | 2026-11-03 | 怎样把逐轮文字反馈用于 OPSD，又保持 RL 稳定？ |
| [On-policy distillation as a Backdoor Propagation Channel](https://arxiv.org/abs/2611.19207) | 2026-11-14 | 教师的条件性异常怎样经软监督传播，怎样抑制？ |

阅读标记约定：**原文事实**指对应版本的叙述、定理或测量；**本文推演**是为理解而整理的代数、教学算例或伪代码，不是复现实验；**未披露/未核实**表示不能从现有证据补全。原图均从对应 v1 PDF 截取，保留来源与图表号；算例均不作为实验图。代码核查截止本次运行，仅 DWE 核实到专用官方仓库；未找到不等于断言不存在。

## 先建立共同的基础概念

**状态和 rollout。** 对语言模型，状态 $`s=(x,y_{\lt t})`$ 是问题与已经生成的前缀；rollout 是继续生成的一条轨迹。OPD 让学生自己生成，再让教师对这些前缀评分，因此比只模仿固定教师答案更贴近学生实际犯错的位置。OPSD 则可让同一个模型在额外提示、反馈或特权信息下充当教师，两边参数相同也不代表条件概率相同。

**KL 的方向。** 统一令 $`p(v)=\pi_\theta(v\mid s)`$ 为学生分布，$`q(v)`$ 为教师分布，$`v\in\mathcal V`$ 为词表中的 token。两种常用损失为：

```math
\begin{aligned}
D_{\mathrm{KL}}(p\Vert q)&=\sum_{v\in\mathcal V}p(v)\log\frac{p(v)}{q(v)},\\
D_{\mathrm{KL}}(q\Vert p)&=\sum_{v\in\mathcal V}q(v)\log\frac{q(v)}{p(v)}.
\end{aligned}
```

前者称 reverse KL，按学生概率加权；后者称 forward KL，按教师概率加权。它们都不是“只比较生成出来的那个词”。全词表损失看的是全部 token 的软概率。符号 $`\theta`$ 是可训练参数，$`\mathcal V`$ 是共享词表，$`\log`$ 为自然对数；softmax 严格正概率下上述式子有限。实际应在 FP32 中用 log-softmax 和稳定求和，避免把很小的概率舍入为零。

**停止梯度和两个分布。** 训练有“状态从哪里来”和“到达状态后用什么损失”两层选择。常见 OPD 实现把本轮采样出的状态视为固定，再对状态上的蒸馏损失求梯度。它不自动包含对整个采样分布的求导。教师的 stop-gradient 同样重要：把教师分支也反传，会改变方法。

**最优传输与耦合。** 耦合是边缘分布固定的一种联合抽样方式。CoRD 耦合两个 token 分布，DWE 耦合正负梯度坐标，LCS 论文则把失败轨迹分配给成功轨迹。三者的对象和约束不同，不能因为都出现 OT 就混成同一个算法。

<a name="paper-cord"></a>

## 1. CoRD：用最少替换补足 OPD 的状态盲区

**原文与身份。** [v1 全文](https://arxiv.org/pdf/2611.07959v1)，Guhan Chen 等，清华大学、哈工大深圳、南京大学、华为相关团队；2026-11-05 首投。讨论的是文本数学推理的 OPD rollout，不是视觉模型或推理服务框架。

### 问题、直觉和原架构图

**原文事实。** 学生只在自己走过的状态上学习，有些教师偏好的解题路径几乎不会被访问。作者在冻结的状态集合上比较发现，训练前教师可达状态上的 KL 是学生可达状态上的约 1.5 至 2.4 倍。降低学生自身状态上的 loss，不一定意味着补上了这些盲区。

![CoRD 原 Figure 1](assets/cord-fig1.png)

图 1-1｜来源：[CoRD v1](https://arxiv.org/pdf/2611.07959v1)，Figure 1，PDF 第 2 页。左侧是学生 rollout 漏掉教师路径的示意，右侧是耦合采样流程。这是原方法架构图，不是额外测量结果。读图时沿学生提议、接受判断、残差替换和同一 KL 监督的顺序看。

**本文直觉。** 好比学生总沿同一条街练习开车，老师只能纠正这条街上的操作。CoRD 让老师偶尔在岔路口引导学生转弯，但尽可能保留学生原本的选择。改变的是练习地点，不是到达地点后采用的评分标准。

### 关键公式、逐符号解释与证明

令 $`p_S(\cdot\mid s)`$、$`p_T(\cdot\mid s)`$ 分别为学生和冻结教师的下一词分布，$`\alpha\in[0,1]`$ 控制偏向教师的程度。原文 Eq.2 的采样目标为几何插值：

```math
q_\alpha(v\mid s)=\frac{p_S(v\mid s)^{1-\alpha}p_T(v\mid s)^\alpha}{Z_\alpha(s)},\qquad
Z_\alpha(s)=\sum_{u\in\mathcal V}p_S(u\mid s)^{1-\alpha}p_T(u\mid s)^\alpha.
```

$`Z_\alpha`$ 是保证总概率为 1 的归一化项；$`u`$ 是求和用的 token 索引。它不是学生、教师概率的算术平均。实现可对加权 log 概率做 softmax。$`\alpha=0`$ 恢复学生采样，$`\alpha=1`$ 恢复教师采样。

先提议 $`a\sim p_S`$，以 $`A(a)=\min(1,q_\alpha(a)/p_S(a))`$ 接受；否则从正残差采样：

```math
\begin{aligned}
\mathrm{TV}(p_S,q_\alpha)&=\frac12\sum_v|p_S(v)-q_\alpha(v)|,\\
r(v)&=\frac{[q_\alpha(v)-p_S(v)]_+}{\mathrm{TV}(p_S,q_\alpha)},\qquad [z]_+=\max(z,0).
\end{aligned}
```

$`\mathrm{TV}`$ 是 total variation distance，$`r`$ 是仅在目标比学生分配更多概率的 token 上有质量的替换分布。分母为零时所有提议都被接受，不调用残差采样。

**原文 Proposition 1，本文展开证明。** 输出 token 为 $`v`$ 的概率有接受和替换两部分。接受部分为 $`p_S(v)A(v)=\min(p_S(v),q_\alpha(v))`$。总拒绝概率等于正差之和，即 TV；乘上 $`r(v)`$ 后得到第二部分。因此：

```math
\begin{aligned}
\Pr(Y=v)&=\min(p_S(v),q_\alpha(v))+[q_\alpha(v)-p_S(v)]_+\\
&=q_\alpha(v),\\
\Pr(Y\ne a)&=\sum_v[p_S(v)-q_\alpha(v)]_+=\mathrm{TV}(p_S,q_\alpha).
\end{aligned}
```

任意边缘分别为学生和目标的联合分布，匹配质量最多为逐 token 最小概率之和，因此不匹配率至少为 TV。该采样规则达到下界，是“替换最少”的严格含义，并不是“训练耗时最少”。

原文保持每个状态上的全词表 reverse KL，只改变状态分布：

```math
\mathcal L_{\mathrm{CoRD}}=\mathbb E_{s\sim d_{q_\alpha}}
\left[D_{\mathrm{KL}}(p_S(\cdot\mid s)\Vert p_T(\cdot\mid s))\right].
```

$`d_{q_\alpha}`$ 表示按目标策略生成轨迹诱导的状态分布。教师仍是 $`p_T`$，不是拿几何插值替代教师作为 loss target。采样分支停止梯度，只通过学生的状态内损失更新；没有用重要性权重把状态分布改回学生分布，因为这会抵消有意进行的覆盖迁移。

### 教学算例与训练实现

**本文推演，非原实验。** 设三词分布学生为 (0.6,0.3,0.1)，教师为 (0.2,0.3,0.5)，取 $`\alpha=1`$，目标就是教师。提议第一词时仅以 1/3 接受，另外两词总被接受。TV=0.4，残差只剩第三词，因此输出第一词概率为 0.2，第二词为 0.3，第三词为 0.1+0.4=0.5。40% 的提议被替换已足够；随意从教师重抽并不能保持同样的最小替换率。

**本文按 Algorithm 1 整理的伪代码：**

```python
for prompts in fixed_prompt_order:
    states = []
    for prefix in autoregressive_generation(prompts):
        ps, pt = student_and_frozen_teacher(prefix)
        q = softmax((1-alpha)*log(ps) + alpha*log(pt))
        a = categorical(ps)
        if uniform() < min(1, q[a]/ps[a]):
            token = a
        else:
            token = categorical(normalize(clamp(q-ps, min=0)))
        states.append(prefix)
        append_token(token)
    loss = mean(full_reverse_kl(student(states), teacher(states)))
    update_student_only(loss)
```

教师、学生必须共享 token ID 语义。实现要保持同一 prefix 的 KV cache 同步，不能在替换后仍沿未接受的学生前缀评分。奖励：没有新增任务 RL 奖励，训练信号是教师分布；数学判分器用于评估。原文未提供专用可运行官方代码，SimpleRL-Zoo 链接属于评估依赖，不能作为 CoRD 实现交付。

### 数据、规模、预算与核心实验

**原文事实。** 固定约 5,000 道 MATH 训练题，固定顺序、两样例 completion 模板；每步 256 prompts，共 80 步，rollout 最多 320 新 tokens，学习率 $`10^{-5}`$。Qwen3 学生 0.6B/1.7B/4B，教师 8B；Qwen3.5 学生 0.8B/2B/4B，教师 9B；另有 14B 教师分析。主实验 $`\alpha=0.6`$。

![CoRD 原 Table 1](assets/cord-table1.png)

图 1-2｜来源：CoRD v1，Table 1，PDF 第 5 页。条件：五个数学基准 MATH500、AIME24/25、AMC23、GSM8K，每题生成 16 次，温度 1、最多 4096 tokens。按学生规模分块，比较 untrained、OPD、CoRD，以及部分规模的 SKD top-25、OPD 160 步。先比较同一学生的 Avg.，再查单项；不是每个单项都提高。

六个学生的平均 pass@1 均提高 0.9 至 2.0 个百分点，pass@8 提高 1.2 至 3.4 个百分点。Qwen3-0.6B 的平均 pass@1 从 10.2 到 12.2；Qwen3.5-2B 从 16.0 到 16.9，训练 OPD 160 步为 15.1。pass@16 并非普遍提高：Qwen3-1.7B 的均值低 0.1 点。AIME 每套仅 30 题，小幅波动不能过度解读。

![CoRD 原 Figure 4](assets/cord-fig4.png)

图 1-3｜来源：CoRD v1，Figure 4，PDF 第 8 页。Qwen3-4B-Base、8B 教师，每 8 步在冻结状态集测量。左/中图分别是教师/学生可达状态的 KL gap closure，越高表示相对初始差距消除得越多；右图是替换率，不能当成成功率。对照 OPD 和原图标示的替换规则。

closure 定义为 $`100(1-\mathrm{KL}_{\mathrm{after}}/\mathrm{KL}_{\mathrm{before}})`$。4B 在 40 步时教师侧 closure 为 21.7%，OPD 为 15.0%；CoRD 在 16 步达到 OPD 80 步仍未达到的教师侧 closure。Table 3 的随机 token、平均词频 token、随机位置教师替换都没有同样效果，说明替换方向重要。Table 4 增大 alpha 改善教师侧 closure，但公共基准 pass@16 在 alpha=1 不及 0.6，不宜简单认为教师介入越多越好。

**成本与边界。** 原比较给所有方法逐 token 同时跑两模型，使每步耗时差异在 2.3% 内；实际标准 OPD 可先独立生成，再一次性教师评分，CoRD 却需要生成时的教师 logits。因此“同一步数公平”不能直接推出生产端到端等成本。未披露完整硬件和总 GPU 时，不能自行补算美元成本。较小学生、数学任务、同 tokenizer、短训练 rollout 限制外推；并未证明一般 agent 环境都收益。

<a name="paper-dwe"></a>

## 2. DWE：全词表 loss 保留，输出层反传做无偏稀疏估计

**原文与身份。** [v1 全文](https://arxiv.org/pdf/2611.27131v1)，Yuyang Zhang、Yuxin Li，中国科学技术大学；2026-11-20 首投。正文重点是离线 KD，但 Appendix B.3 直接评估 OPD，本期主要分析这部分及其共同估计器。

### 问题和原方法图

**原文事实。** 输出层矩阵 $`W\in\mathbb R^{V\times d}`$ 把隐藏向量 $`h\in\mathbb R^d`$ 映射为 logits $`z=Wh`$。全词表反传需要算隐藏状态梯度和权重梯度，词表大时开销明显。DWE 不截掉 loss 的小概率项，而是允许每个位置的梯度带噪声，只要求期望保持正确。

![DWE 原 Figure 1](assets/dwe-fig1.png)

图 2-1｜来源：[DWE v1](https://arxiv.org/pdf/2611.27131v1)，Figure 1，PDF 第 3 页。按 a 正负梯度拆分、b 固定几何排序、c 共享分层分位数、d gather/scatter 反传的方向阅读；图中 K=2。它画的是梯度坐标采样，不是 token rollout 的采样。

### 从 softmax 到无偏估计的推导

**原文 Eq.2/3 与 Appendix A.4。** 令 $`g=\nabla_z\ell`$。forward KL 有 $`g_i=p_i-q_i`$；OPD 使用 reverse KL，不能直接套用这个残差。对 reverse KL，先有 $`\partial\ell/\partial p_j=\log(p_j/q_j)+1`$，再代入 softmax Jacobian $`\partial p_j/\partial z_i=p_j(\mathbf1_{i=j}-p_i)`$，得到：

```math
\begin{aligned}
\ell&=\sum_jp_j\log\frac{p_j}{q_j},\\
g_i&=p_i\left(\log\frac{p_i}{q_i}-\ell\right),\qquad \sum_i g_i=0,\\
\nabla_h\ell&=W^\top g,\qquad \nabla_W\ell=gh^\top.
\end{aligned}
```

$`i,j`$ 为词表坐标，$`W^\top`$ 为矩阵转置；$`\ell`$ 是当前位置的 scalar loss。零和性质也可从“给所有 logits 加同一常数，softmax 不变”推出。它让正负梯度可以配对守恒。

原文 Eq.4 至 7 将 $`g^+_i=\max(g_i,0)`$、$`g^-_i=\max(-g_i,0)`$ 分开：

```math
\begin{aligned}
m&=\sum_i g_i^+=\sum_i g_i^-=\tfrac12\lVert g\rVert_1,\\
r_i^+&=g_i^+/m,\qquad r_i^-=g_i^-/m,\\
I&\sim r^+,\quad J\sim r^-,\quad \widehat g_{\mathrm{pair}}=m(e_I-e_J),\\
\mathbb E[\widehat g_{\mathrm{pair}}]&=m(r^+-r^-)=g.
\end{aligned}
```

$`m`$ 是正负两侧共同总质量；$`r^+,r^-`$ 是概率分布；$`e_I`$ 是第 I 坐标为 1 的标准基向量。只写两个坐标就保持期望正确。$`g=0`$ 时返回零估计，避免除零。每对都有一正一负，避免独立抽样把预算随机浪费在同号坐标上。

**几何为何重要？** 边缘分布决定无偏性，但联合分布决定乘过 W 后的方差。令 $`w_i`$ 是 W 第 i 行的转置，原 Eq.8 为：

```math
\mathbb E\left\lVert m(w_I-w_J)-W^\top g\right\rVert_2^2
=m^2\mathbb E\lVert w_I-w_J\rVert_2^2-\lVert W^\top g\rVert_2^2.
```

展开平方后，中间交叉项借无偏性化简为负的真梯度范数。第二项固定，因此让成对输出行靠近，可降低隐藏状态梯度 MSE。原文不在每个 token 求昂贵高维 OT，而是初始化时中心化 W、估计主成分、按投影排序为置换 $`\tau`$，以后缓存这个顺序。共享同一个分位数，分别在正负 CDF 中查坐标。

为减少分位数集中，原文再把 (0,1) 分成 K 层，每层均匀抽一个 $`U_k`$：

```math
\begin{aligned}
U_k&\sim\mathrm{Unif}\left(\frac{k-1}{K},\frac{k}{K}\right),\\
I_k&=\tau(F_{+,\tau}^{-1}(U_k)),\quad
J_k=\tau(F_{-,\tau}^{-1}(U_k)),\\
\widehat g&=\frac mK\sum_{k=1}^K(e_{I_k}-e_{J_k}),\\
\mathbb E\lVert\widehat g-g\rVert_2^2
&\le\frac{2m^2-\lVert g\rVert_2^2}{K}.
\end{aligned}
```

$`F_{+,\tau},F_{-,\tau}`$ 是排序后正负分布 CDF；$`F^{-1}`$ 指累计概率查找；$`k`$ 是层编号，K 是 pair 数。每层单独不必服从完整分布，K 层平均才恢复原积分，因此不能用固定中点替代随机 jitter 还宣称同一无偏证明。

参数梯度也保持条件无偏：固定状态与参数，若 $`J_\theta=\partial z/\partial\theta`$，则 $`\mathbb E[J_\theta^\top\widehat g\mid s,\theta]=J_\theta^\top g`$。这不是保证每次更新、最终模型、训练随机过程都与全梯度相同；Adam 的非线性状态和采样方差仍会影响轨迹。

### 教学算例、伪代码和官方代码检查

**本文推演。** 令零和 logit 梯度为 (0.3,0.1,-0.2,-0.2)，则 m=0.4，正侧概率 (0.75,0.25)，负侧两坐标各 0.5。一对可能抽到第一与第三坐标，估计为 (0.4,0,-0.4,0)，本次误差明显，但重复平均的第一坐标期望为 0.4×0.75=0.3。若只保留最大绝对值坐标，丢掉的项期望无法恢复。具体 pair 后的隐藏梯度误差还取决于这两行 W 的距离。

```python
order = principal_component_order(initial_student_output_head)
for student_rollouts in training:
    p, q = aligned_full_vocab_distributions(student_rollouts)
    kl = sum(p * (log(p)-log(q)), axis=-1)
    g = p * ((log(p)-log(q)) - kl[..., None])
    positive, negative = ordered_positive_negative_parts(g, order)
    for k in range(K):
        u = (k + uniform()) / K
        i = inverse_cdf(positive, u)
        j = inverse_cdf(negative, u)
        accumulate_sparse_pair(i, j, positive_mass/K)
    backward_via_gathered_rows_and_scatter_add()
```

这是教学伪代码，未替代数值精度、mask、重复索引累加、教师 detachment 或 fused kernel 的实现。数学中两侧质量相同；浮点下官方实现分别保留两侧实际 mass，避免强制相等引入偏差。训练仍计算完整概率及 dense output-weight gradient，不能宣传为整个训练都变成稀疏内存。

**已核实代码。** [官方 DWE](https://github.com/dwe-lab/DWE)，固定提交 [481223f](https://github.com/dwe-lab/DWE/tree/481223f3704607b2c9cb21741bfd3fde8822299b)，Apache-2.0。已阅读 README、sampling.py 和 geometry.py：quantile 分支按层加随机数、共享正负分位数；缓存 PCA 排序，固定 RNG 并处理 PCA 符号；还保留 independent/coupled/coordinate/topk 控制分支。代码阅读不代表跑过 H100 实验。README 明确示例数据为手写 smoke，实际数据、权重与教师 cache 需另行准备。

### OPD 配置、实测速度和消融

**原文事实，Appendix B.1/B.3。** 学生 Qwen3-1.7B、教师 Qwen3-4B-Instruct-2507，关闭 thinking；约 100K prompts，等量来自 OpenMathReasoning、OpenCodeReasoning、OpenScienceReasoning-2，仅用问题，响应在线生成。1×H100 80GB，100 updates，global batch 256，microbatch 2、累积 128，最大响应 8192、温度 0.7；学习率峰值 $`5\times10^{-6}`$、终值 $`5\times10^{-7}`$、10 步 warmup，cosine decay；ZeRO-2、BF16 projection/FP32 probabilities，K=8 即每 token 16 坐标抽取，seeds 42/43/44。

![DWE 原 Figure 4](assets/dwe-fig4.png)

图 2-2｜来源：DWE v1，Figure 4，PDF 第 22 页。OPD 输出层计时，以 Full 归一化；1K/4K/8K 是被测 token 位置数，柱子拆分 forward、估计准备、backward。三领域平均，先 30 次 warmup 再 100 次交错测量；一次几何排序不计入。它不是完整训练速度图。

输出层总加速 1.96 至 2.20 倍；8K 位置时从 214.6ms 到 97.7ms。不要把 8.6 至 20.2 倍 backward 局部加速也称为完整更新加速。

![DWE 原 Table 10](assets/dwe-table10.png)

图 2-3｜来源：DWE v1，Table 10，PDF 第 23 页。上半是三种 seed 的完整更新秒数，下半是阶段剖面；逐阶段采样仅两个 warmup 后三次更新，不能要求其总和等于完整训练均值。对照是同设置 Full reverse KL。

完整 OPD 更新均值 Full=688.54s、DWE=678.03s，节省 **1.53%**。原文指出 generation 和 teacher backbone 占 Full 剖面约 53.2%，解释局部优化为何收益被稀释。Table 11 的 13 基准平均：Full 38.95、IS 39.18、DWE 39.30；“相近”比“显著优于”更稳妥，因为各基准评估次数和方差不同。

![DWE 原 Figure 5](assets/dwe-fig5.png)

图 2-4｜来源：DWE v1，Figure 5，PDF 第 24 页。三列分别为三个训练 seed，上排 throughput、下排 reverse KL，每点汇总五次更新。Full 和 DWE 在各自响应上测量，loss 接近不是在完全相同状态分布上的逐项等价。

**原文 Table 3 消融。** 1,024 个固定 KD token 位置，K=8、相同16坐标预算，128 次抽样平均，relative MSE 为平方误差除以真梯度平方范数：

| 配置 | Hidden MSE | Head MSE |
| --- | --- | --- |
| IS | 0.3042 | 0.1679 |
| 独立正负配对 | 0.2019 | 0.1293 |
| 几何共享 IID 配对 | 0.1941 | 0.1283 |
| 几何共享分层配对 DWE | 0.1110 | 0.0707 |

这些测量说明本设置下最大追加收益来自分层，但不能据此保证每个词表、每步训练都有相同比例改善。

**原文消融与范围。** 同抽样预算下，DWE 相对 IS 的隐藏状态/输出权重梯度 MSE 下降 63.5%/57.9%，来自正文梯度估计实验；配对、几何排序、分层各控制不同的方差来源。离线 KD 的 10.4% 至 15.7% GPU 时节省、1.63 至 1.95 倍输出层加速，条件是 80M FineWeb-Edu tokens、缓存教师 hidden states且排除缓存构建。不能移植为本段 OPD 的端到端节省。OPD 增量输出层峰值内存只降 4.5% 至 6.0%，不是整个模型显存减半。

**本文补充复杂度解释。** N 个 token、词表 V、隐藏宽度 d 时，原文 Eq.16/17 的两项 dense backward 乘加合计为 $`2NVd`$，DWE 至多 $`4NKd`$，算术量比为 $`V/(2K)`$。这是这两个乘法的算术量比例，不是实际时钟速度；全词表前向、扫描 CDF 和 scatter 的内存访问仍要付费。

**局限/复现性。** 需要完整教师概率访问与共享输出 token IDs；只降低输出层部分 backward，不消除全词表 forward/估计准备。初始化的几何顺序可能随训练变旧；低 K 下依然有估计方差。论文提供 H100、种子、精度、超参和公开实现，但本期没有训练复现、没有重测其时钟，也不从“条件无偏”推断“收敛完全一致”。奖励：本 OPD 运行无额外任务奖励，优化 reverse KL。

<a name="paper-lcs"></a>

## 3. Which Successes Make Good References?：成功不等于适合纠错

**原文与身份。** [v1 全文](https://arxiv.org/pdf/2611.13135v1)，Mufei Li 等，Georgia Tech 与 AWS；2026-11-09 首投。作者主页的 11 月 12 日 news 是公告时间，不覆盖 arXiv 首投时间。

### 多轮路径依赖和方法图

**原文事实。** 参考增强 OPD 把一条同任务成功轨迹给教师，学生仍只看自己的历史。但成功方案可以走不同房间、操作不同对象或修改不同数据库记录。教师若拿不兼容方案纠错，可能破坏学生已经做对的选择。原文四次教师采样中，有成功参考的任务里，ALFWorld 的 33%、airline 的 82.8% 至少有两条不同成功轨迹，选择并非虚设。

![LCS 原 Figure 1](assets/lcs-fig1.png)

图 3-1｜来源：[该文 v1](https://arxiv.org/pdf/2611.13135v1)，Figure 1，PDF 第 2 页。两条成功参考的最长公共前缀均为 3，但 LCS 分别为 3 和 5；选择共享有序行为骨架更多的参考。右侧强调参考仅进入教师，OPD loss 作用于失败学生 rollout，部署不需要成功参考。

**基础概念。** POMDP 中真实环境状态不可见，历史 $`h_t`$ 汇集 observations 和已执行 actions。LCP 是从第一步开始连续相同的长度；LCS 是可以跳过若干中间动作、但保持顺序的最长共同子序列。相同字符串片段不必表示相同实际状态，LCS 是可观察代理而非状态恢复器。

### 自由目标 OT 为什么只需逐条找最近参考

**原文 Eq.1/Proposition 3.1。** 设任务 x 的 N 条失败轨迹为 $`\tau_i^-`$，成功集合为 $`\mathcal R_x`$。以 $`\gamma_{ij}`$ 表示分给成功参考 j 的失败质量，$`c_{ij}`$ 为纠正成本。目标参考使用频率不固定，因此可写为：

```math
\min_{\gamma_{ij}\ge0}\sum_{i,j}\gamma_{ij}c_{ij},\qquad
\sum_j\gamma_{ij}=\frac1N.
```

$`i`$ 枚举失败，$`j`$ 枚举同任务成功；只有失败端质量约束，没有要求每条成功平均使用。**本文展开推导：** 每一行至少支付 $`(1/N)\min_j c_{ij}`$；把该行全部质量给最小成本列就达到下界，故整体最优为每条失败独立选择最佳参考。若 $`c=-s`$，则：

```math
\tau^*(x,\tau^-)
\in\arg\max_{\tau^+\in\mathcal R_x}s(\tau^-,\tau^+).
```

这不是需要 Sinkhorn 迭代的固定两端 OT；也不保证所有成功轨迹都会被用到。集合为空时回退无参考 OPD。

**原文 LCS 实例化。** 把轨迹变成已执行动作序列，最长相同有序子序列长度记为 $`\ell_{\mathrm{LCS}}`$，分数为：

```math
s_{\mathrm{LCS}}(\tau^-,\tau^+)
=\frac{\ell_{\mathrm{LCS}}(\tau^-,\tau^+)}{Z(T^-,T^+)}.
```

$`T^-,T^+`$ 是两序列长度，Z 为长度归一化。原文 Appendix D 对 ALFWorld 取两者最小长度，对 tau3-bench 取失败长度；二者不能随意互换。

**本文推演，动态规划实现。** 令 C(i,j) 是前 i 个失败动作、前 j 个成功动作的 LCS 长度；边界行列为零：

```math
C(i,j)=\begin{cases}
C(i-1,j-1)+1,&a_i^-=a_j^+,\\
\max(C(i-1,j),C(i,j-1)),&a_i^-\ne a_j^+.
\end{cases}
```

它来自最后动作相同时接在前一最优序列后，不相同时至少跳过一侧。单候选时间 $`O(T^-T^+)`$，仅求长度可用滚动行降低内存；动作 JSON 的参数顺序及实体标准化需确定，否则字符串比较会制造不必要失配。论文未附专用可运行代码，不能确认其所有 tie-break 与规范化边角。

### 理论条件、教学例子和损失

**原文 Theorem 3.1/3.2。** 在同初始状态、同确定性转移且未匹配动作块可恢复到相同状态的条件下，匹配动作对应一致状态。关键是“可恢复”，不能仅由 LCS 大就无条件推出。证明按匹配位置归纳：初始状态相同，未匹配块按假设重新汇合，随后执行相同匹配动作，由相同转移得到下一共同状态；重复此过程。若缺少汇合假设，这个归纳链就断开。若 m 个匹配回合监督收益总计至少 gm，剩余回合总损失最多 b(T−m)，则平均增益下界为：

```math
\frac1T\sum_{t=0}^{T-1}\Delta_t
\ge(g+b)\frac mT-b,\qquad
\frac mT>\frac b{g+b}\ \Longrightarrow\ \text{positive lower bound}.
```

$`\Delta_t`$ 为相对无参考教师、在理想纠正分布下的期望 log 概率增益；g>0 为匹配处收益下界、b≥0 为其他处损失界、T 为失败回合数。推导只是将 gm−b(T−m) 除以 T。原文没有证明实践中的 g、b 已被估计并满足假设。

**本文算例，非实验。** 失败动作是 [开柜、查看、重复查看、取杯、放杯]；参考甲是 [开柜、查看、取杯、放杯]，参考乙是 [开柜、查看、关柜、移到别处]。两者 LCP 均为 2，LCS 分别为 4、2。按失败长度归一化为 0.8、0.4。若重复查看不改变环境状态，甲更可能提供兼容纠正；若那一步实际改变物体位置，则这个直觉的状态假设可能失效。

**原文训练机制。** 先离线每训练任务采四条教师轨迹，只保留奖励验证成功者；在线生成学生轨迹，仅失败者更新。参考进入教师条件 $`\pi_\phi(u_t\mid h_t,\tau^*)`$，学生条件仍为 $`\pi_\theta(u_t\mid h_t)`$。目标是这些状态上的 reverse KL；文中也给出 sampled log-ratio 近似。实现 sampled 估计时必须用正确策略梯度/代理目标，不能把采样动作当常数后只对 log-ratio 随意反传并宣称完整 reverse KL 梯度等价。

```python
reference_pool = four_teacher_rollouts_per_training_task()
reference_pool = keep_reward_validated_successes(reference_pool)
for task in online_batches:
    trajectory, reward = student_interact(task)
    if success(reward):
        continue
    candidates = reference_pool[task]
    ref = highest_normalized_lcs(trajectory, candidates) if candidates else None
    teacher = score_own_history_with_teacher_only_reference(trajectory, ref)
    update_student_with_reverse_kl_on_failed_trajectory(teacher)
```

奖励用于成功筛选和失败 gating，不是新增 LCS reward；LCS 选择参考而非把学生行为强行编辑成参考答案。

### 实验、消融、数据与成本

**原文事实。** ALFWorld：Qwen2.5-3B/7B-Instruct 学生，GiGPO 训练的 7B 教师；训练 200 configured updates，global batch 64 turns，rollout batch 16 trajectories，学习率 $`10^{-6}`$，最多 30 actions。tau3 airline：Qwen3.5-2B/4B 学生、9B 教师，Claude Opus 4.7 用户模拟器；10 configured updates，global batch 15 turns，最多 200 benchmark steps。airline 训练、retail 仅迁移评估，不再优化。

ALFWorld Valid Unseen：3B 从 vanilla OPD 的 78.4% 到 LCS 的 85.8%，7B 从 83.6% 到 94.0%。基线包括 Random、Shortest、BM25、Qwen3-Embedding-0.6B、Edit Distance、LCP、BRTS；参考增强方法共享参考池并采用 failure-only，vanilla 主基线更新所有 rollout，因此主表有两项变化，应配合消融判断。

![LCS 原 Table 2](assets/lcs-table2.png)

图 3-2｜来源：该文 v1，Table 2，PDF 第 7 页。airline 是域内，retail 是不再优化的迁移测试；每任务四次交互，报告三 seed 平均。每个学生独立比较 vanilla OPD 与七种选择器和 LCS。表中 Pass^4 是四次全成功，Pass@4 是至少一次成功，符号差一处但含义差很多。

2B airline Pass^4：30.0%→41.7%；4B：26.7%→38.3%。2B airline 的 Pass^1 和 Pass@4 是 BM25 略高，不能声称 LCS 占据所有单项第一；LCS 是唯一在所测两个学生、两个任务体系全部主聚合指标上都胜过 vanilla OPD 的选择器。4B airline Pass^1 从 51.2% 到 65.4%；retail Pass^1 两学生分别 70.4%→78.3%、56.2%→69.4%。

![LCS 原 Table 3](assets/lcs-table3.png)

图 3-3｜来源：该文 v1，Table 3，PDF 第 8 页。小学生组中拆开是否更新成功 rollout；成功列的勾表示也更新成功样本。LCS 在 ALFWorld Valid Unseen 上从包含成功时的 72.4% 到 failure-only 的 85.8%，说明参考兼容与更新门控需共同核查。附录 Table 9 给出大模型对应消融。

**成本与复现。** 原文 Appendix J 披露 8×A100 80GB，分配为 4 training、2 teacher scoring、2 model update；ALFWorld 约 10 小时，airline 约 1 小时，全体训练/评估约 4,500 A100 GPU-hours。这里保留作者的阶段命名，不猜测双阶段资源共享。按单次 8 卡全占用粗乘 80/8 GPU-hours 只是本文预算理解，不替代原总预算。方法基于 TCOD、Trinity-RFT、veRL、Ray、PyTorch；核查作者主页仅列论文，没有确认专用代码与独立作者教程。

**局限。** LCS 不是隐藏状态一致性的充分观测证据；API 可非确定、动作名相同但参数或上下文不同、不可逆副作用都会破坏可恢复假设。候选池受教师成功率限制，fallback 覆盖不能省略。模型规模、两个 benchmark、短训练和模拟用户限制泛化；无真实业务部署成本证明。

<a name="paper-guidance"></a>

## 4. Targeted Guidance：只在失败回合给方向，同时保留 RL 奖励

**原文与身份。** [v1 全文](https://arxiv.org/pdf/2611.05266v1)，Kevin Frans、Alexander M. Rush、Sergey Levine、Pieter Abbeel、Ashvin Nair；Cursor Research 与 UC Berkeley。首投 11 月 3 日，11 月 4 日更新为 v2，本节固定 v1。

### 为什么纯奖励可能学不会

**原文事实。** coding agent 对某些新工具格式或文字风格，几乎没有成功探索，单一末端奖励只告诉它好坏，难以告诉它具体怎么改。作者研究的目标不是把 coding pass-rate 本身称为新能力提升，而是在保住主要任务表现时学习这些难以优化的辅助准则。

![Targeted Guidance 原 Figure 2](assets/tg-fig2.png)

图 4-1｜来源：[该文 v1](https://arxiv.org/pdf/2611.05266v1)，Figure 2，PDF 第 3 页。失败 turn 前加入文字提示，教师以同一权重读取额外反馈，对学生原有 token 给分布级指导。黄色反馈分支不是部署时输入，局部 KL 不作用到所有回合。

**本文直觉。** 奖励像给作文打分，反馈像指出“这一段需要描述修改结果”。只给低分可能反复写不出来；每一段都重复纠正又可能扰乱已经做好的内容。方法把定向提示限在具体失败位置，并让 RL 继续负责保留和改善整体行为。

### 目标函数和策略改进条件

原文把总奖励写为 $`R(\tau)=\sum_i\lambda_i R_i(\tau)`$。i 是准则，$`R_i`$ 是任务通过或风格/工具准则，$`\lambda_i`$ 是奖励权重，不要与后面的蒸馏权重混淆。某回合 t 未达准则时给反馈 $`f_t`$，构造 $`\mu(\cdot\mid s_t)=\pi_\theta(\cdot\mid s_t,f_t)`$，在本次更新内停止教师梯度。

原文 Eq.1/2 是设计启发而非实践保证：若反馈策略在当前状态分布下优势为正，且与原策略足够接近，朝它小步移动有利：

```math
\begin{aligned}
\mathbb E_{s\sim d_\pi,a\sim\mu}[A^\pi(s,a)]&>0,\\
\mathbb E_{s\sim d_\pi}[D_{\mathrm{KL}}(\pi(\cdot\mid s)\Vert\mu(\cdot\mid s))]&\lt \delta.
\end{aligned}
```

$`A^\pi`$ 是动作相对当前策略状态价值的优势；$`d_\pi`$ 是当前策略状态分布；$`\delta`$ 是希望控制的散度尺度。文字提示未必构成更好的策略，因此原文先比较有无提示的生成表现，仅允许平均改进的提示进入训练。

原文 Eq.3 将 GRPO 与失败回合上的全词表 reverse KL 相加。按正文“跨整个 rollout token 数归一化”说明展开，可以写为：

```math
\begin{aligned}
\mathcal L(\theta)&=\mathcal L_{\mathrm{GRPO}}(\theta)
+\frac{\lambda}{N(\tau)}\sum_{t\in\mathcal F}\sum_{j=1}^{n_t}
D_{\mathrm{KL}}\bigl(p_{t,j}\Vert\mathrm{sg}(q_{t,j})\bigr),\\
p_{t,j}&=\pi_\theta(\cdot\mid h_t,y_{t,\lt j}),\\
q_{t,j}&=\pi_\theta(\cdot\mid h_t,f_t,y_{t,\lt j}).
\end{aligned}
```

这是**按原 Eq.3 与文字说明展开的本文记号**。$`\mathcal F`$ 是失败回合集，$`n_t`$ 是回合 token 数，$`N(\tau)`$ 是 rollout 全部有效 response tokens，$`\lambda`$ 是蒸馏权重，sg 表示 stop-gradient。提示只插在对应回合前，非失败位置 mask 为零；分母不是失败 token 数。padding 不应计入 N。

**本文推演。** 若只有 1 个失败 token，其 KL=2，而 rollout 有 100 个有效 token，则未乘 lambda 的辅助损失为 0.02；若误除失败数量会变成 2，压力放大 100 倍。所有准则满足时辅助项消失，剩下原 RL 目标；这不保证有限训练中已达到全局最优。

为理解基础 GRPO，可把一组 G 条 rollout 的分数转为 $`\widehat A_i=(R_i-\bar R)/(\sigma_R+\epsilon)`$，用新旧策略概率比与 clipped surrogate 更新。此处是**基础概念**，不是声称论文给出了独特 GRPO 变体或完整异步校正实现。RL reward、KL weight、生成长度限制必须一起调；论文的异步推理策略可能落后于训练策略。

### 训练流程、实现细节和原实验

```python
feedback_templates = validate_prompts_against_contextless_policy()
for rollout_group in async_coding_agent_rollouts:
    task_scores, criterion_scores = judge(rollout_group)
    rl_loss = ordinary_grpo(task_scores, criterion_scores, length_control=True)
    mask, feedback = failed_turns_only(criterion_scores)
    targets = same_weights_no_grad_forward(original_tokens, feedback)
    kd_loss = masked_full_vocab_reverse_kl(targets, mask)
    kd_loss /= all_valid_response_tokens(rollout_group)
    update(rl_loss + kd_weight * kd_loss)
```

**原文事实。** KL 在 trainer 侧计算整个词表，避免 inference/trainer mismatch 的 sampled 估计问题；同时包含 thinking 和 tool-call tokens。主要新增开销是一遍教师 forward，不要求教师自回归重写整轮回答。Online SFT 等比较方法会生成新教师回合，成本和行为干预不同。

数据是多轮软件工程题，单 epoch、同一个 prompt 只训练一次；Qwen3-Coder-30B 经 agent harness SFT 初始化，200 步，batch 为 32 groups、每组 32 rollouts，Adam、学习率 $`3\times10^{-6}`$。论文未公开完整私有题集、训练硬件/GPU 时及完整超参 sweep 清单，不能算出确定美元成本。原文依赖链接 quack 不是该方法端到端代码，尚未核实专用官方训练仓库。

![Targeted Guidance 原 Table 1](assets/tg-table1.png)

图 4-2｜来源：该文 v1，Table 1，PDF 第 6 页。比较相同 coding RL 基础上的辅助风格方法，最后一列为主要 coding pass-rate；误差为最后 32 步标准差，不能当多随机种子置信区间。Baseline 仅优化 coding；GRPO 增加辅助奖励；OPSD 只增加反馈蒸馏；还有 Online SFT、LaTRO、Verifree。

Targeted Guidance 的 coding pass-rate 为 0.33±0.05，baseline 为 0.33±0.05，OPSD 为 0.30±0.05。方法在 Markdown、approval、命令输出等准则上分别达到 0.85、0.55、0.64；纯 GRPO 的 Markdown 和 approval 仍为 0。应读成该设置中保留主任务同时学到辅助准则，不是 coding 能力全面大幅跃迁。

![Targeted Guidance 原 Figure 5](assets/tg-fig5.png)

图 4-3｜来源：该文 v1，Figure 5，PDF 第 7 页。每回合诗歌状态更新准则；左图准则达成，右图 coding pass-rate；比较 Baseline、GRPO、OPSD、Targeted Guidance 与 always-on 反馈。看左图会觉得 OPSD 也会学诗，右图显示稳定性代价；always-on 也伤及主要任务。不要把这一更长曲线实验的横轴范围与 Table 1 的 200 步配置混写。

![Targeted Guidance 原 Figure 6](assets/tg-fig6.png)

图 4-4｜来源：该文 v1，Figure 6，PDF 第 8 页。不同 distillation weight，依次显示 pass-rate、style reward、entropy、token length。高权重初期学风格快，随后 entropy 和长度尖峰与 pass-rate 崩落相伴；原文未证明所有尖峰都具有同一因果机制。

![Targeted Guidance 原 Figure 7](assets/tg-fig7.png)

图 4-5｜来源：该文 v1，Figure 7，PDF 第 8 页。比较只蒸馏 message 与同时蒸馏 thinking/message，四个风格准则面板显示后者更少振荡。thinking 参与监督是该实验的稳定性结论，不应推广成所有模型和任务一律必须显示思维链。

**局限与复现。** 反馈依赖手工静态字符串、judge 和局部可修改准则；“可以独立改风格而不伤代码”的假设未必适用于互相冲突的真实目标。论文提到 forward KL、JS 等效果近似，但未深入系统比较，不能据此宣称方向无关。缺公开 harness/完整数据/训练成本限制逐数复现；本期只做原文、图表与机制核查。

<a name="paper-teacher-risk"></a>

## 5. Backdoor Propagation Channel：把教师软监督也纳入可信度检查

**原文与身份。** [v1 全文](https://arxiv.org/pdf/2611.19207v1)，Zane Fu 等，Alibaba 与合作高校；2026-11-14 首投。本节分析 OPD 风险机制与防御，不提供投毒训练脚本或真实系统利用流程。

### 研究问题、威胁模型与原图

**原文事实。** 假设教师已经存在条件性异常行为，下游学生做正常在线蒸馏，训练问题流中有少量触发条件暴露。作者问：即使没有直接给学生异常 target response，甚至过滤了学生 rollout 中的异常回答，教师全词表概率是否仍能把该条件行为传过去？

![Backdoor OPD 原 Figure 1](assets/risk-fig1.png)

图 5-1｜来源：[该文 v1](https://arxiv.org/pdf/2611.19207v1)，Figure 1，PDF 第 2 页。上层是研究中的不可信教师来源，下层是学生生成、教师分布监督的 OPD 通道。关键区分是输入暴露和软概率监督；论文不声称没有触发输入也能在此实验中传播。

**本文直觉。** 老师在学生“暂时还没写出的答案”上也分配概率。只检查实际生成文本，可能漏掉老师悄悄提高某类 token 支持度的信号。另一方面，条件概率不同本来也是正常语义理解的基础，不能把所有差异都视为异常。

### 核心公式、梯度机制与数值例子

原文 Eq.2 仍为学生状态上的全词表 reverse KL；不额外使用学生恶意标签或任务 reward。对同语义的普通/条件输入，在同一 response prefix 下，教师分布分别为 $`q_C,q_T`$，原 Eq.5 定义支持变化：

```math
\delta(v)=\log\frac{q_T(v)}{q_C(v)}.
```

$`C,T`$ 分别是 clean 和 triggered 条件，$`v`$ 为 token。为单独隔离教师变化，**原文固定同一个学生分布 p**，而不是把两种输入下自然不同的学生也混进来。令 $`\ell(p,q)=\lambda D_{\mathrm{KL}}(p\Vert q)`$，则 Eq.7 至 9 为：

```math
\begin{aligned}
\Delta\ell&=\ell(p,q_T)-\ell(p,q_C)\\
&=\lambda\sum_vp(v)\left[\log\frac{p(v)}{q_T(v)}-\log\frac{p(v)}{q_C(v)}\right]\\
&=-\lambda\sum_vp(v)\delta(v),\\
\frac{\partial\Delta\ell}{\partial z_v}
&=-\lambda p(v)\left[\delta(v)-\sum_up(u)\delta(u)\right].
\end{aligned}
```

第一步取消两项 log p，第二步用 softmax Jacobian 得到中心化差异。$`z_v`$ 是学生 logit，$`\lambda>0`$ 是损失权重。某 token 的教师支持变化高于学生加权平均时，梯度差为负，相对 clean 监督更倾向抬高该 logit。这里计算的是两种教师监督的**梯度差**，不是声称总梯度由这一项独占。

**本文安全教学算例。** 只设两个无害标签 A/B：学生 (0.5,0.5)，普通教师 (0.5,0.5)，条件教师 (0.8,0.2)，lambda=1。delta 约为 (0.470,-0.916)，平均为 -0.223；两坐标梯度差为 (-0.347,+0.347)。即使本次没有抽到 A，完整 soft distribution 仍可能产生偏向 A 的更新。这说明概率通道，不是新的攻击实验或泛化 ASR 预测。

**原文机制干预。** 把成对教师 log 概率拆成共享均值 $`M=(\ell_C+\ell_T)/2`$ 和半差 $`\Delta=(\ell_C-\ell_T)/2`$。Removed 对两种条件都用 log-softmax(M)；Misdirected 保留差异大小但打乱一半成对方向。若条件行为主要靠稳定的差异方向传播，两种干预都应削弱条件分离。这比只看相关性更接近受控验证，但仍局限于给定威胁模型。

### 防御：改写对照与监督差异门控

**原文 Eq.10 至 12。** 先对可能造成条件敏感的低频词进行保持任务语义的对照改写为 $`x^R`$，学生始终在原输入 x 上生成。教师对原/改写输入的同一 prefix 分别给出 $`q_k,q_k^R`$；令中间分布 $`M_k=(q_k+q_k^R)/2`$，用 JS 散度决定监督目标：

```math
\begin{aligned}
s_k&=\tfrac12D_{\mathrm{KL}}(q_k\Vert M_k)
+\tfrac12D_{\mathrm{KL}}(q_k^R\Vert M_k),\\
\widetilde q_k&=\begin{cases}
q_k^R,&s_k>\tau,\\
q_k,&s_k\le\tau.
\end{cases}
\end{aligned}
```

$`k`$ 为 response token 位置，$`\tau`$ 为门限，不是轨迹；$`\widetilde q_k`$ 替代原教师进入 reverse KL。关键并非删掉学生输入或生成新教师答案，而是在监督进入更新之前做条件差异检查。改写可能破坏语义，须先验证任务目标及语义一致，不能把新分布自动视作“干净真值”。

**本文按防御侧整理的伪代码：**

```python
for prompt in approved_training_prompts:
    control = semantically_checked_rewrite(prompt)
    response = student_generate(prompt)
    q = teacher_score(prompt, response_prefixes(response))
    qr = teacher_score(control, response_prefixes(response))
    divergence = js_divergence(q, qr)
    target = where(divergence > threshold, qr, q)
    update_student(full_reverse_kl(student_logits(response), stop_grad(target)))
```

该伪代码只解释防御接口，未运行或验证具体门控效果。额外 teacher scoring、改写生成/检查会增加成本；原文未给出完整门控端到端时延和 GPU 时。

### 原实验、消融和边界

![Backdoor OPD 原 Figure 3](assets/risk-fig3.png)

图 5-2｜来源：该文 v1，Figure 3，PDF 第 5 页。四列是 Llama-3.2-3B→3B、3B→1B、Qwen2.5-7B→1.5B、Qwen3-32B→8B；横轴为触发输入比例。上排普通 OPD，下排先过滤异常 rollout；看 ASR、FTR 和两者差值，不能只盯 ASR。下排仍出现条件分离，支持 soft supervision 通道假说。

**原文事实。** 0.5% 输入暴露时，不同配置 ASR 31.7% 至 71.7%；2% 时最高 89.2%，并非所有模型同一数值。固定 2%、Llama-3.2-3B→3B 的 Table 1 中，OPD ASR 89.2±8.8、offline reverse KL 53.3±2.9、sequence KD 56.7±10.4、vanilla forward KL KD 61.7±7.6。训练轨迹来源、KL 方向和监督形式分别变化，不能仅归因 KL 方向。

![Backdoor OPD 原 Figure 5](assets/risk-fig5.png)

图 5-3｜来源：该文 v1，Figure 5，PDF 第 7 页。四种架构、2% exposure，比较 Original、Removed、Misdirected；上排 ASR/FTR，下排 Conditional Gap。Removed 和打乱方向都削弱 gap；某些设置 gap 变负表示方向反转，不等于“正常任务一定无损”。原 Figure 6 的表示投影与行为同步仅作辅助机制证据，不视作独立因果证明。

![Backdoor OPD 原 Table 2](assets/risk-table2.png)

图 5-4｜来源：该文 v1，Table 2，PDF 第 9 页。Llama-3.2-3B→3B、harmful-content 场景、2% exposure、门限 0.0025；对照 undefended、ONION、Fine-mixing、Fine-pruning、Clean fine-tuning。下文只转述防御指标，不把此设置推广到所有教师来源。

门控后 ASR 从 89.17±8.78 降至 15.00±2.50，FTR 从 15.83±13.77 降至 5.00±2.50，gap 从 73.33±11.81 降至 10.00±4.33。ONION 的 gap 为 9.17，略低于本方法，但 ASR/FTR 都更高；因此不能笼统声称本方法在每个指标都第一。附录 E 门限 sweep 表明过小门限伤及正常能力、过大门限过滤不足，0.0025 是该设置折中点。

**数据与预算。** 原文以 Alpaca 构造基础任务，另按四类目标行为构造评估；OPD 每次约 10K input-only prompts，一输入采一轨迹，温度/top-p 均为 1，150 updates、global batch 64、学习率 $`10^{-5}`$，prompt/response 上限为 256/128；BF16 训练、FP32 概率/KL。教师构造与学生蒸馏是不同阶段，不能把教师异常标签记成直接给学生的训练目标。未披露明确硬件数量、总 GPU 时或官方可运行代码；本期不补造训练成本或代码可复现性。

**局限。** 条件词需要一定输入暴露，并假设能找到保语义对照；多轮、长上下文、更大模型、多模态尚未系统验证。输出过滤未挡住此通道，不意味着任何过滤都无用；门控亦无对所有异常教师的一般保证。安全 judge 使用 claude-opus-4-6，评估依赖 judge 误差和目标定义。关键实践启发是审计 teacher supervision 的稳定性与语义一致性，而不只检查训练文本。

## 跨论文比较与基础资料

| 方法 | 修改哪个环节 | 保留什么 | 主要代价或风险 |
| --- | --- | --- | --- |
| CoRD | rollout 状态分布 | 状态内全词表 reverse KL | 生成过程需逐 token 教师 logits |
| DWE | 输出层反传估计 | 完整 soft loss、条件无偏期望 | 估计方差、仍有全词表前向 |
| LCS references | 教师参考选择与 failure-only | 学生原历史、部署不读参考 | 状态兼容代理可能失效 |
| Targeted Guidance | 失败回合反馈加辅助 KL | 主要 coding RL 与各准则奖励 | 大权重造成熵/长度尖峰 |
| Difference Gated Distillation | 教师监督的条件差异 | 原输入上的学生 rollout | 改写质量与额外教师评分成本 |

**本文综合判断。** 这些方法分别解决覆盖、梯度计算、监督上下文、学习稳定和监督可信度，不是可直接拼接后保证收益的套件。比如 CoRD 改变状态分布后，DWE 在固定状态上的无偏性仍可讨论，但端到端速度和方差要重测；门控改写本身也可能改变教师纠错方向。没有联合实验时，不把组合写成已验证方案。

辅助资料均不占五篇名额：

- [GKD/On-Policy Distillation 原始论文](https://arxiv.org/abs/2306.13649)：理解自生成状态上的蒸馏及状态分布错配。
- [Speculative Sampling 原始论文](https://arxiv.org/abs/2302.01318)：理解接受/残差采样为何保持目标分布；CoRD 借用其耦合机制，不等同报告解码加速。
- [PyTorch KLDivLoss 官方文档](https://docs.pytorch.org/docs/2.15/generated/torch.nn.KLDivLoss.html)：核对输入是 log 概率、target 的表示与 reduction；不要从 API 参数顺序猜 KL 方向。
- [PyTorch searchsorted 官方文档](https://docs.pytorch.org/docs/2.15/generated/torch.searchsorted.html)：理解 DWE 的 inverse-CDF 查找及边界。
- [TRL GRPO 官方教程](https://huggingface.co/docs/trl/main/en/grpo_trainer)：理解分组优势、采样与更新；main 版本会变化，本期不声称等于论文运行环境。
- [DWE 作者代码与教程](https://github.com/dwe-lab/DWE/tree/481223f3704607b2c9cb21741bfd3fde8822299b)：README、docs/theory.md、docs/experiments.md、docs/evaluation.md，可核查实现和评估协议。
- [Mufei Li 作者主页](https://mufeili.github.io/)：核对 LCS 论文与公告；它不是一份独立技术博客。其余四篇未核实到独立作者博客，不拿第三方解读冒充。

检索与排除记录：软提示 CSPD（2611.27376）使用 verifier-filtered 教师离线响应及 NLL，不计入本期直接 OPD/OPSD 名额；U-OPSD（2608.06296）首投在窗外，不以新播客发布日期替代；旧论文的普通更新、已在其他方向收录项不重复。论文原文、日期快照、代码提交、原图裁剪出处与文件哈希见 manifest.json 和 assets/SOURCES.md。

数学排版说明：全文 Markdown 的行内公式采用 GitHub 支持的受保护 LaTeX，独立公式采用 math fence；同一正文生成 PDF，保留所有文字、公式和原图。PDF 独立公式为矢量排版，行内公式按文字基线对齐；assets/math-manifest.json 记录公式源码与渲染核查信息。本期未进行模型训练复现，教学数值均明确标注，不伪装为实测曲线。
