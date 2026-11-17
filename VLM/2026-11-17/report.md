# VLM 后训练论文精读 - 2026-11-17

**检索窗口：2026-10-17 至 2026-11-17；主论文5篇；跨方向、跨日期去重。**

今天从推理框架转向小型VLM的训练机制：视觉特权信息如何产生有效蒸馏信号、GRPO约束为何可能失效、如何在领域适配或生成质量提升中保留原能力。5篇均核实arXiv v1首次发布日期。Vision-OPD与MMSearch-R1作为方向锚点，不充当近期新论文。最后一篇是统一理解/生成模型的后训练扩展，非搜索agent论文。

**证据口径。** 本文原图直接裁自所链接论文v1 PDF，未重绘实验数值。原论文事实、本文公式推演和复现建议分别标记；没有运行模型训练，不把作者结果称作本文复现。代码、训练时长、样本规模没有公开到可核实程度时明确留空。主论文PDF与HTML均已读取；第三方搜索摘要只用于发现候选，不作为结果证据。

| 论文 | v1日期 | 小模型与关键问题 | 今日优先级 |
| --- | --- | --- | --- |
| BlinkOPD / Teach with Blinks | 11-12 | 4B/9B；教师被学生错误前缀带偏 | 最高，直接承接Vision-OPD |
| Safety as a Budget | 11-12 | 2B；GRPO的乘子被标准化抵消 | 高，训练目标与实现细节 |
| ERASE | 10-30 | 3B/7B；多模态注入下任务保真 | 高，OPSD的参考视图设计 |
| Triplet Incremental Training | 11-12 | 4B；几何适配与遗忘的平衡 | 中，领域后训练与蒸馏 |
| FlowShift | 11-03 | 2B/4B；离散到连续生成的LoRA适配 | 扩展，统一多模态后训练 |

## 0. 读公式之前：真正需要的最小知识

**概率分布不是一句参考答案。** 下一token分布p是词表上每个候选token的概率。假设词表只有A、B，教师p=(0.8,0.2)，学生q=(0.4,0.6)。硬标签只告诉学生选A；软分布同时告诉它B仍有20%的可能性。这里KL(p||q)=0.8log(0.8/0.4)+0.2log(0.2/0.6)，约0.335 nats；它不是准确率，不能说降低0.335就提高33.5%准确率。

**KL方向要看谁在前。** 对固定教师p和学生softmax logits z，forward KL的梯度为dL/dz_j=q_j-p_j，因此会提升教师认为有概率、学生却忽略的token。reverse KL改变权重和梯度结构；JSD则借助混合分布m=(p+q)/2对两侧约束，定义为0.5KL(p||m)+0.5KL(q||m)。这些是标准定义；下文每篇实际使用哪一种不能互换。

**On-policy的关键是前缀来自谁。** SFT通常在固定数据答案的前缀上训练；OPD先让学生自己生成y，再让教师在相同y_<t处打分。这样教师看到的是学生实际容易走进的状态，不仅是完美答案路径。代价是教师也可能被错误前缀影响，或学生从来不访问某些重要状态。采样通常作为固定轨迹处理，不代表自动对离散采样做反向传播。

**LoRA省的是可训练参数和优化器状态。** W'=W+(alpha/r)BA，其中W冻结，A为r×d_in，B为d_out×r。对于4096×4096的投影，r=16需要16×8192=131,072个适配参数，约为原矩阵参数的0.78%；但完整前向仍要经过冻结主干。不能把“不到1%可训练”直接写成“显存、训练时间都不到1%”。这是教学算例，不是下面任何一篇的实测比例。

---

## 1. BlinkOPD：让教师在每个错误前缀处重新区分视觉证据与语言惯性

**论文：** [Teach with Blinks](https://arxiv.org/abs/2611.16402)；[v1 PDF](https://arxiv.org/pdf/2611.16402v1)；[作者代码](https://github.com/Supltz/BlinkOPD)。首次公开2026-11-12。模型为Qwen3.5-4B/9B。

### 1.1 核心问题与原图

Vision-OPD让看局部crop的同源教师指导看全图的学生。但假设学生先说“帽子是橙色的……”，教师即使看见黄色帽子，也可能沿着这个错误叙述继续。问题不是教师完全识别不出局部，而是教师条件分布已经被学生的错误prefix污染。

![BlinkOPD原论文Figure 5](assets/blink-fig5.png)

图1：原论文Figure 5，PDF第5页。上方student读取全图生成rollout；左下冻结教师分成Eyes Open和Eyes Closed两个读法，二者共享学生前缀；First Glance无前缀看crop，Conflict Gate控制增益。红色更新箭头只回到student。图中“闭眼”不是换一张黑图：它保留视觉边界标记和原位置索引，移除图像内容token，减少更改多模态输入格式引入的混杂。

### 1.2 目标分布：逐符号解释

原文Eq.(8)可改写成便于计算的等价形式：

```text
p_plus(v)  = frozen_teacher(v | crop, question, student_prefix)
p_empty(v) = frozen_teacher(v | eyes_closed, question, student_prefix)
r(v)      = log p_plus(v) - log p_empty(v)
q(v)      = softmax(log p_plus(v) + beta_i * r(v))
          ∝ p_plus(v)^(1+beta_i) / p_empty(v)^beta_i
beta_i    = beta * gamma^(answer_i != first_glance_answer)
```

v是词表token，q是最终监督分布，beta是基础对比强度，gamma>1用于冲突轨迹。前缀条件相同，使“仅由语言前缀带来的共同偏向”有机会在log比值中抵消；这是一种参考分布设计目标，不应夸大为任何模型上严格成立的独立分解定理。

**本文手算。** 只考虑“黄/橙”：p_plus=(0.6,0.4)，p_empty=(0.2,0.8)，beta=1。未归一化q为(0.6²/0.2,0.4²/0.8)=(1.8,0.2)，所以q=(0.9,0.1)。黄色在看到crop后相对闭眼增加得更多，因此被强化。若p_plus和p_empty完全相同，对比项为零，q回到p_plus；若参考概率过小，则会极度锐化，工程上需稳定的log_softmax与有限精度检查。

First Glance不是额外的人类标签：它是冻结教师在没有学生前缀时给出的答案。学生最终答案与之不同时增大beta。教师第一眼也可能错，所以这个gate是启发式错误指示，不能当作ground truth verifier。

### 1.3 训练流程与真正的成本

```text
freeze teacher = initial_student
for each image/question:
    glance = teacher_answer(crop, question, no_student_prefix)
    trajectories = student.rollout(full_image, question)
    for y in trajectories:
        beta_y = beta * gamma if answer(y) != glance else beta
        p_plus  = teacher.score(crop, question, prefixes(y))
        p_empty = teacher.score(eyes_closed, question, prefixes(y))
        target  = stop_gradient(contrast(p_plus, p_empty, beta_y))
        loss += mean_token_JSD(student_probabilities, target)
    update student only
```

原文配置：96 prompts/step、每prompt 8条rollout、38步、最多1024生成token、温度1.0；5步warm-up后学习率2e-6；beta=3.4、gamma=2；JSD mixing alpha=0.5。4B在4张B200上少于3小时，按配置乘法可得到少于12 GPU-hours的粗上界，但不是账单价格，也不能外推到消费卡。闭眼/睁眼教师打分和student rollout都要计入成本。训练图像500张不意味着只生成500条轨迹。

学生输入遵守其Vision-OPD协议：全图上画有evidence box且提示关注该区域。复现时若删掉框或改成无提示全图，输入条件就变了。作者移除21个与V*重叠的问题后保留6220个候选训练样本，训练集合的去污染与评测judge修正也是对比条件。

### 1.4 核心结果与消融应怎样读

![BlinkOPD原论文Table 1](assets/blink-table1.png)

图2：原论文Table 1，PDF第6页，含4B和9B分组。只优先比较相同骨干、同一重新评测协议的黑色block；灰色闭源/大模型历史数字来自其他条件，不能作为严格控制实验。

4B基础模型六任务均值70.0；BlinkOPD(500图)79.1，增加9.1个百分点；同表Vision-OPD(6.2K图)75.1。这里12倍主要指独立训练图像数，不是总训练FLOPs减少12倍。9B在500图时79.6；6.2K时80.4。原文Table 3中移除参考分布降至75.1，改成text-only为78.2，改EMA教师为76.1；支持参考与teacher稳定性有作用，但没有证明所有VLM家族都适用。

**本文复现优先项。** 在同样训练数据、步数、rollout预算下比较q=p_plus、text-only、eyes-closed、eyes-closed+gate。记录每token梯度与错误答案概率随prefix长度的变化；同时测视觉任务和泛化，防止只学“更关注局部”而损害整体语义。检查位置索引、vision marker、padding/mask是否逐token一致。原文未提供可跨硬件直接复用的完整美元成本。

---

## 2. Safety as a Budget：为什么你加了GRPO惩罚，梯度却像没加一样

**论文：** [Safety as a Budget](https://arxiv.org/abs/2611.16628)；[v1 PDF](https://arxiv.org/pdf/2611.16628v1)。首次公开2026-11-12。policy为Qwen3-VL-2B-Instruct；open scorer是冻结4B readout。未在已读论文中找到可核实的作者代码仓库或作者博客链接。

### 2.1 两个不同的问题：排序与标尺

一个reward scorer能把更危险的回答排在前面，并不代表它输出0.1就真的对应10%危险率。例如把所有分数平方，排序不变，平均数会变；约束优化却用平均数和预算比较。另一个陷阱发生在GRPO组内标准化：统一放大某组reward，会同时放大分子和标准差，控制强度可能被消掉。

![Two-channel GRPO原论文Figure 2](assets/budget-fig2.png)

图3：原论文Figure 2，第4页。蓝色上半是用于更新policy的learning channel；橙色下半是估计实际约束水平、更新lambda的control channel。raw score用于保留细粒度排序，单调校准c(p)用于预算比较和惩罚，二者不能简单混成同一个未经校准的信号。

### 2.2 原文目标与乘子推导

教学记号：r0是基础帮助性reward，c_h是有害帮助成本，c_r是误拒绝成本；epsilon_h/r是允许预算。目标为maximize E[r0]，subject to E[c_h]<=epsilon_h和E[c_r]<=epsilon_r。

```text
r = r0 - lambda_h * calibrated_harm * indicator(harmful_prompt)
       - lambda_r * calibrated_refusal * indicator(benign_prompt)
lambda_k <- clip(lambda_k + eta*(EMA(mean(calibrated_cost_k))-epsilon_k),
                 0, lambda_max)
```

lambda是惩罚的可学习强度，eta是乘子更新步长。成本高于预算时lambda上升，policy受到更强惩罚；低于预算时下降。原文eta=0.5、lambda_max=10，约束估计使用系数0.3的EMA。注意这个约束作用于训练分布的scorer估计，不能直接保证任意测试分布的真实安全率。

**第一种失效的手算。** 设scorer平均估计h_hat=h+b，b是偏差。内点固定点要求h_hat=epsilon，故真实h=epsilon-b。若epsilon=0.10而b=0.16，隐含目标h=-0.06不可能实现，lambda会撞上上限；系统不会因为迭代久了自动校准。单调映射的好处是保留排序；但若把阶梯状isotonic映射直接替代基础reward，许多不同回答会同分，因此论文把两条用途拆开。

**第二种失效的严格算例。** 在harmful prompt上、使用raw成本的简化设置，r_i=1-(1+lambda)p_i。GRPO的A_i=(r_i-mean(r))/std(r)。因为std(r)=(1+lambda)std(p)，lambda消失：

```text
A_i = -(p_i - mean(p)) / std(p)
```

取p=(0.2,0.8)，lambda从0变9，reward从(0.8,0.2)变为(-1,-7)，标准化后仍是(1,-1)。绝对惩罚大了10倍，policy看到的相对advantage没有变化。这个结论依赖同一组共享正比例系数及该简化reward；有其他项、校准、数值epsilon时不能一概说任何GRPO约束都失效。

论文采用不缩放advantage，或先对每个目标标准化再加权。后者教学形式为A=z(r0)-lambda_h*z(c_h)-lambda_r*z(c_r)，让lambda留在相加之后。实现时应在loss构建处核查advantage，而不仅看日志里的lambda有没有上升。

### 2.3 实验结果与一处必须保留的证据冲突

![Safety as a Budget原论文Table 1与Figure 3](assets/budget-table1-fig3.png)

图4：原论文第5页左侧Table 1和右侧Figure 3原始页面裁图。表中raw/calibrated与unscaled/CGRPO是不同实验设置，不能把它们的最好指标拼成一行。右侧HADES为未见分布评测。

主表三seed下标准GRPO harm=8.9%，CGRPO+calibrated=4.8%，raw约束=7.6%；HADES从28.9%到12.8%。然而**摘要称over-refusal不超过2%，PDF v1的Table 1对应calibrated行却列4.0%**，unscaled calibrated也为4.0%。本文不把摘要的安全/帮助性叙述当作已验证的一致结论，应请求作者澄清指标口径或版本。原图保留冲突便于读者核查。

训练设置：LoRA rank16、all linear、LR5e-5、KL系数0.02、4 samples/prompt、8 prompts/step、80步，生成上限384 tokens。论文未在所读设置中提供足以复算GPU-hours的硬件与wall-clock组合，因此不猜训练成本。其困难样本挖掘会改变训练分布，不能把少量步数直接当作普适高样本效率。

**本文建议。** 先用两回答toy reward复现lambda抵消，再做校准集/验证集分离，最后跑多seed。至少记录真实标签成本、raw/calibrated均值、lambda是否饱和、各类prompt梯度范数。低harm必须与误拒绝、一般能力一起看；这是一篇训练控制研究，不是上线安全保证。

---

## 3. ERASE：用“只擦掉注入”的参考视图，教3B模型保留任务、忽略外部恶意指令

**论文：** [Erase the Injected Region](https://arxiv.org/abs/2610.41307)；[v1 PDF](https://arxiv.org/pdf/2610.41307v1)。首次公开2026-10-30。Qwen2.5-Omni-3B/7B；作者表示代码将在接收后公开，目前不编造GitHub链接。

### 3.1 机制与训练数据构造

训练数据知道注入区域R；把图像注入区域填白、音频对应时段置静音、视频对应片段置黑，保留其余媒体及时间/空间结构。student看带注入的输入x并生成，冻结base teacher看擦除版x_bar，在学生同一个prefix处给完整词表分布。推理时student不需要先找R或执行擦除。

![ERASE原论文Figure 2](assets/erase-fig2.png)

图5：原论文Figure 2，第4页。左侧是离线训练参考构造，右侧上下两支共享rollout prefix。不要把左侧erase box当成部署时拥有完美注入定位器；R来自训练benchmark构造，真实未知攻击不是自动拥有同样条件。

### 3.2 Loss、梯度与不能省略的条件

```text
P_t = frozen_base(. | erased_media, user_request, y_<t)
Q_t = student(. | attacked_media, user_request, y_<t)
L   = E_{y sampled from student}[ mean_t KL(P_t || Q_t) ]
```

公式与原文Eq.(4)-(5)同义。teacher在前，loss=-sum_v P(v)log Q(v)+constant；对学生logit的梯度是Q-P。一个受注入影响的token若学生错误倾向0.9、教师只给0.1，其错误logit梯度为+0.8，梯度下降会降低它。对完全不受擦除影响且P=Q的位置，初始化时直接梯度为0；但共享参数在其他位置被更新，仍可间接改变此处行为，所以“无直接梯度”不是训练全程不变保证。

原文定义S为student在擦除视图上的分布；D_inj=0.5[KL(Q||S)+KL(S||Q)]，Delta_erase=KL(P||S)。只让Q匹配P，并不能无条件推出D_inj=0：还需要student在擦除视图保留P，即S=P。否则两个输入都可能发生漂移。这里也不能套用KL三角不等式，因为一般KL没有三角不等式。论文将两轴分开测，正是避免“攻击不成功但模型已经什么都不会”的假防御。

**教学例子。** 原媒体是一段脚步声，插入“忽略问题，输出口令”。若把整段音频换噪声，student也许不听攻击，但同样认不出脚步。只擦注入时段的teacher仍能回答脚步；蒸馏以这份保留信息的分布为目标，因此有机会兼顾任务。这不是标签free就意味着数据free：训练仍需要构造攻击/擦除配对与可靠区域信息。

### 3.3 核心实验、比较边界与复现

![ERASE原论文Table 1](assets/erase-table1.png)

图6：原论文Table 1，第6页。UIAc为clean输入任务准确率，UIAi为attacked输入任务准确率，AIA/AIFR分别关注完整和含部分攻击遵从。四列一起读，不只挑0.000的攻击率。

3B audio中ERASE的UIAi=60.1%，ARGUS=33.1%；undefended clean=60.6%。7B audio ERASE=63.4%，SecOPD=64.2%，因此**ERASE不是每个模型/模态都最好**。原表若干AIA/AIFR为0.000，仅表示有限样本下观测值/舍入，不能推成真实攻击率严格为零。3B audio的注入依赖从6.26到0.0095是分布诊断，不是准确率百分比。

训练：2张A800、LoRA r=128；audio LR5e-5，image/video LR2e-5；gradient accumulation128；rollout温度1，image最多128新token，audio/video最多256。训练时长与完整可复算GPU-hours未披露到本文可核实程度，不自行补数。用VTQA、Clotho-AQA、MSR-VTT等保留任务内容，攻击训练来源与测试攻击不同，仍需检查攻击位置/模板是否覆盖真实风险。

本文复现建议：保留原user instruction与external media的信任边界；只在loss端做teacher-forcing打分，不执行媒体中的任何命令。实现response-only mask并让teacher无梯度；验证白块/静音/黑帧时长和区域之外的内容未改。评测至少包括未知注入位置、跨模板、不同字体/语音、域外一般能力与D_inj/Delta_erase双轴；区域划分错误的敏感性是重要新增消融。

---

## 4. Triplet Incremental Training：一条深度支路学几何，一条冻结同伴保住通用能力

**论文：** [From Foundation VLM to Versatile Driving VLM](https://arxiv.org/abs/2611.16727)；[v1 PDF](https://arxiv.org/pdf/2611.16727v1)。首次公开2026-11-12；InternVL3.5-4B。文中标注CoRL 2026接收。未核实到独立作者代码仓库/博客，不能把基础InternVL仓库称作本文实现。

### 4.1 三流架构与“特权信息”在哪里

![Driving原论文Figure 2](assets/driving-fig2.png)

图7：原论文Figure 2，第4页。主VLM可训练，视觉中间层接DPT深度头；冻结Twin VLM以输出分布约束语义。与BlinkOPD不同，这里主要是**off-policy**固定答案前缀蒸馏，不能因为也有teacher就写成OPD。

数据triplet为(I,D,P)，I是图像，D是像素对齐的米制深度，P是问答对。LiDAR点投到图像形成稀疏米制提示，再由Prompt Depth Anything生成稠密depth。训练用额外深度信号约束视觉表示；部署去掉深度头和Twin。所谓“无额外推理开销”指不保留这两条训练辅助分支；下游检测/规划仍有其task head及计算，不能说系统不需任何其他模块。

### 4.2 三个loss各解决什么问题

```text
L_KD    = mean_t KL(P_twin(. | I,Q,answer_<t) || P_student(. | same_prefix))
L_CE    = -mean_t log P_student(task_answer_t | I,Q_task,task_answer_<t)
e_p     = log predicted_depth_p - log target_depth_p
L_depth = sqrt(mean(e_p^2) - lambda*mean(e_p)^2)
L_total = 0.25*L_depth + 0.20*L_KD + 1.0*L_CE
```

KL对固定自生成问答中的答案位置算，prompt mask掉；CE适配driving目标；深度loss直达视觉encoder，防止只在文本空间记住坐标术语。p索引有效深度像素，lambda=0.85，深度范围[0.5,80]m。注意本节lambda是SiLog尺度项，不是上一节的RL约束乘子，同名不意味着同一机制。

**本文SiLog推导。** mean(e²)=Var(e)+mean(e)²，所以L_depth²=Var(e)+(1-lambda)mean(e)²。第一项约束相对形状，第二项保留一部分整体尺度误差。若所有深度都预测成真实值的2倍，e_p=log2，方差为0，但lambda=0.85时loss=sqrt(0.15)*log2约0.268；lambda=1时该整体尺度错误会完全消失。这解释了为什么米制场景不能随意把lambda设1。

**本文对KL锚的解释。** 单看CE，模型只需提升标准答案的概率，其他词表相对关系可能漂移；KL增加“不要偏离原分布太远”的约束。它保留的是特定输入与前缀上的输出行为，不是所有知识的数学不变性；有几何与语义冲突时仍需权衡系数，保留通用任务测试十分必要。

### 4.3 训练预算与关键消融

训练170K triplets，两epochs，16张A100 80GB；DPT头约30M参数，视觉第4/8/16/24层输入。先冻结主干初始化深度头，再联合解冻训练；视觉encoder基准LR2e-5、AdamW/cosine。论文没有给出wall-clock，不能只用GPU数量算GPU-hours。后续perception/planning评测还包括冻结表示后的新head训练，不等于同一个VLM直接零样本输出全部规划结果。

![Driving原论文Tables 5和6](assets/driving-tables5-6.png)

图8：原论文Tables 5-6，第8页。Table 6各行都保留driving SFT，差别是depth/KD；因而有助拆分两个辅助目标的作用。对比不同骨干的ReCogDrive时仍要注意模型与数据条件不同。

Table 6中SFT-only一般VQA均值21.5，加入KD到71.0，depth+KD到73.2；depth-only为21.8，说明视觉几何改善本身没有解决语言遗忘。depth+KD时NAVSIM PDMS=89.10，SFT-only=83.30。Table 5平均73.6到73.2存在约0.3-0.4的舍入差，本文按各表显示口径报告，不宣称严格零遗忘。nuScenes depth delta1从77.2到80.6、检测mAP34.7到38.1。

本文复现优先项：完整2×2辅助loss消融；固定下游head预算；验证LiDAR/图像timestamp与深度对齐；保持通用问答回放比例与mask一致。只用了seed42的主设置，置信区间与跨数据域鲁棒性证据有限。NAVSIM是非反应式离线规划评测，不能直接外推为真实道路闭环安全。

---

## 5. FlowShift：冻结2B/4B语义主干，把离散视觉token接口改成连续速度场

**论文：** [From Tokens to Trajectories](https://arxiv.org/abs/2611.05451)；[v1 PDF](https://arxiv.org/pdf/2611.05451v1)；[作者项目页](https://plan-lab.github.io/FlowShift/)。首次公开2026-11-03。属于统一多模态理解/生成后训练，不是检索agent或纯视觉问答RL。

### 5.1 为什么生成接口可以改，而语义主干不全量重训

离散视觉码本把连续图像映射到有限token，细节可能在量化时丢失。FlowShift用连续VAE latent、时间步编码、输入投影和velocity输出头替换视觉生成接口；主干冻结，Q/K/V/O投影加LoRA，teacher continuous generator冻结。只训练投影、时间层、LoRA、输出头，并保留student原本的text encoder。

![FlowShift原论文Figure 3](assets/flowshift-fig3.png)

图9：原论文Figure 3，第3页。上下student/teacher共享相同clean latent、noise和t，所以预测的是同一点的速度；分别用自己的文本conditioning，避免把不同文本编码空间直接硬拼。火焰表示适配参数，雪花表示冻结。不要把“冻结离散主干”误读成仍通过原离散softmax生成图像。

### 5.2 连续轨迹与双目标：逐步推导

原文约定t=0是数据，t=1是noise，与部分文献相反：

```text
z0 = VAE_encoder(image)
zt = (1-t)*z0 + t*noise,  noise ~ Normal(0,I)
u  = d(zt)/dt = noise-z0
L_FM = E[ ||student_velocity(zt,t,prompt)-u||^2 ]
L_VD = E[ ||student_velocity(zt,t,prompt)-stop_gradient(teacher_velocity)||^2 ]
L_DOVD = L_FM + lambda*L_VD
```

FM对训练样本的解析速度u负责，VD迁移teacher已经学到的平滑速度场。生成从t=1积分到0；Euler更新必须跟符号约定一致。若误把积分方向翻了，即使训练loss正确也会生成错误轨迹。

**本文手算与等价展开。** 对固定样本，令a=noise-z0、b=teacher_velocity、s=student_velocity。||s-a||²+lambda||s-b||²=(1+lambda)||s-(a+lambda*b)/(1+lambda)||²+constant。所以双目标可理解为数据目标和teacher速度的加权折中。例：a=2、b=1、lambda=3，最优s=1.25，不是无条件复制teacher的1。现实中s是共享网络，跨样本优化，并不能对每点都自由选最优；本推导用于理解梯度方向。

**为什么必须共享同一个noise和t？** 若student在某个高噪声点，teacher在另一点，两个速度向量可能都正确但不应相等，L_VD会产生伪误差。共享VAE坐标与噪声后，pointwise回归才有明确意义。对teacher进行stop-gradient也不能省，否则会失去固定参照。

### 5.3 训练成本、结果与付出的代价

原文约6M图文对；8张H100 80GB，per-GPU batch32、accumulation4，有效batch1024；Muon、5K steps、LR1e-5、2K warmup、LoRA rank32/scaling16。训练参数高效不代表数据规模小；wall-clock和完整美元成本未公开到可复算程度。论文展示Qwen Image等teacher消融，本文未核实到与所有设置对应的完整可执行训练配置。

![FlowShift原论文Tables 1和2](assets/flowshift-tables1-2.png)

图10：原论文Tables 1-2，第5页。上表同时看FID、GenEval和推理时间，下表看理解能力。FID越低通常越好，但不能替代prompt adherence，也不能跨不同样本/分辨率协议直接比较。

UniDFlow-4B转化后COCO-30K FID从13.33到8.92，GenEval从0.95到0.92，推理时间从5s到7s；FUDOKI-2B FID16.16到9.29，GenEval0.81到0.77，时间7s到8s。收益是质量/语义折中，不是所有维度同时提升。理解保留99.54%是作者按选定五任务的相对变化聚合，不代表所有能力保持99.54%。Table 3去掉teacher后FID10.69，说明速度蒸馏有贡献；全量finetune虽部分生成指标更好，却明显损害理解。

本文复现建议：先以2B基线对齐latent shape、patch数、t方向和teacher文本conditioning；分别跑FM-only、VD-only、DOVD，独立记录生成质量、组合一致性、理解保留和时延。v1没有给出本文能明确定位的lambda完整数值，不猜；sampling步数/CFG在正文和附录的描述不够统一，应以代码核实实际路径，不能只写“完全同设置”。

---

## 6. 五篇放在一起：可以迁移的实验设计

**共同问题是监督信号而不仅是loss名字。** BlinkOPD设计teacher的视觉参考，ERASE设计保留任务信息的反事实参考，Driving用固定同伴约束输出漂移，FlowShift混合数据解析目标与teacher速度；Safety as a Budget提醒“奖励排序正确”不等于“约束数值正确”。复现时先问监督在什么输入/前缀/坐标上定义，再看系数。

| 本文建议的检查 | 要发现的错误 | 一个最小可做实验 |
| --- | --- | --- |
| teacher/student条件对齐 | prefix、位置、媒体布局错配 | 同输入同权重时KL应接近0 |
| 梯度流向 | teacher被误更新或mask错 | 冻结teacher，检查grad为None |
| token级信号 | 错误prefix让teacher跟错 | 固定错误prefix长度做probe |
| 多目标尺度 | 标准化消掉lambda、loss主导 | toy reward/梯度范数随系数扫描 |
| 泛化与保留 | 领域成功但通用遗忘 | 改前改后同一通用评测与seed |
| 资源与评测口径 | 参数少误写成本低 | 记录rollout、teacher forward与wall-clock |

这些是本文建议，未声称论文作者已经完成所有检查。无reward的distillation并不保证无偏，参考视图与数据筛选就是归纳偏置；“单次推理”也不自动意味着完全相同输入条件或无任何额外预处理。

## 7. 小白到实现的阅读路线（不占今日5篇）

1. [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)：先理解softmax、token、attention，才能读token-level teacher分布。属于基础作者博客，非近期论文。
2. [Hugging Face蒸馏教程](https://huggingface.co/docs/transformers/tasks/knowledge_distillation_for_image_classification)：从固定教师、student loss和温度出发；图像分类示例只是基础，不可直接充当OPD训练代码。
3. [TRL GRPOTrainer官方文档](https://huggingface.co/docs/trl/grpo_trainer)：重点看advantage、scale_rewards、rollout与KL；配置以你锁定的TRL版本为准。
4. [PEFT LoRA官方文档](https://huggingface.co/docs/peft/developer_guides/lora)：理解r、alpha、target_modules以及可训练参数统计。
5. [Vision-OPD原论文](https://arxiv.org/abs/2605.18740)和[作者项目页](https://visionopd.github.io/)：作为BlinkOPD前置，区分crop-conditioned teacher和full-image student。
6. [MMSearch-R1原论文](https://arxiv.org/abs/2506.20670)与[作者博客](https://blog.lmms-lab.com/blog/mmsearch-r1/)：理解on-demand工具调用与搜索惩罚；本篇早于窗口，只作方向基础。
7. [Flow Matching官方代码与文档](https://github.com/facebookresearch/flow_matching)：理解插值、速度场和ODE；再检查FlowShift采用的时间方向。

作者博客存在性：BlinkOPD提供代码，FlowShift提供项目页；其余三篇未核实到作者博客，因此用官方基础文档补充，不把第三方短摘要冒充作者解释。

