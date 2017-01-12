# LDAGibbsSampling
Yet another Python implementation of collapsed Gibbs sampling for Latent 
Dirichlet Allocation 
(http://en.wikipedia.org/wiki/Latent_Dirichlet_allocation),
as described in

	Griffiths, T. L., & Steyvers, M. (2004). Finding scientific topics. 
	Proceedings of the National Academy of Sciences, 101, 5228-5235

Full text: http://www.pnas.org/content/101/suppl.1/5228

---

三种类型的主题模型：LSI，PLSI，LDA。LSI为$A_{m*n}=U_{m*k}*\Sigma_{k*k}*V_{k*n}$，其中A代表m个term、n篇文章，U代表m个term、k个topic，V代表k个topic、n篇文章。开源库[gensim](https://github.com/mahatmaWM/gensim)均有实现。

PLSI，LDA讲解的比较清楚的博文[通俗理解LDA主题模型](http://blog.csdn.net/v_july_v/article/details/41209515)，[LDA主题模型可视化分析](http://cpsievert.github.io/LDAvis/reviews/vis/#topic=1&lambda=0&term=)。

关于lda的评测[LDA主题模型的评估](http://blog.csdn.net/pipisorry/article/details/42460023)，了解Perplexity(混乱)与熵的定义，用熵来定义Perplexity，其值越小越好（其值越小说明对应的熵也越小，集合处在一种比较有序的状态，模型较好）。

---

理解lda参数训练的内幕。
第一步，理解蒙特卡洛方法，其作用是在计算机中得到任意分布的一个样本（一般采样次数越多越准确）。核心思想是使用随机数（或更常见的伪随机数0-1均匀分布）来解决一些复杂的计算问题。
第二步，理解马尔科夫链以及矩阵的极限等概念，马氏链要收敛，但是采样多少次才会收敛，在工程实践中我们更多的靠经验和对数据的观察来指定 Gibbs Sampling 中的 burn-in 的迭代需要多少次。[随机采样和随机模拟：吉布斯采样Gibbs Sampling](http://blog.csdn.net/pipisorry/article/details/51373090)，[MCMC](http://www.ctolib.com/topics-105669.html)

---

todo

周末整理代码的时候，重点参考这个网页。http://blog.csdn.net/yangliuy/article/details/8302599
em与gibbs的区别？？
EM算法的一般步骤为：

1. 随机选取或者根据先验知识初始化；
2. 不断迭代下述两步
①给出当前的参数估计，计算似然函数的下界
②重新估计参数θ，即求，使得
3. 上述第二步后，如果收敛（即收敛）则退出算法，否则继续回到第二步。

[概率语言模型及其变形系列](http://www.52nlp.cn/%E6%A6%82%E7%8E%87%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B%E5%8F%8A%E5%85%B6%E5%8F%98%E5%BD%A2%E7%B3%BB%E5%88%97-lda%E5%8F%8Agibbs-sampling)
[自然语言处理工具包spaCy介绍](http://www.52nlp.cn/)
