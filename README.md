<script type="text/x-mathjax-config">
  MathJax.Hub.Config({
    extensions: ["tex2jax.js"],
    jax: ["input/TeX", "output/HTML-CSS"],
    tex2jax: {
      inlineMath: [ ['$','$'], ["\\(","\\)"] ],
      displayMath: [ ['$$','$$'], ["\\[","\\]"] ],
      processEscapes: true
    },
    "HTML-CSS": { availableFonts: ["TeX"] }
  });
</script>
<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>
# 备忘

要点：doc目录中八卦那篇文章对lda所涉及到的数学知识做了一个总结（以及这个网页版本[LDA-math-文本建模](http://cos.name/2013/03/lda-math-text-modeling/)），另外一篇讲gibbs采样的分析也很详细。
本代码的实现参考[论文](http://www.pnas.org/content/101/suppl_1/5228.full.pdf)

其他：
理解gibbs lda参数训练的内幕。
第一步，理解蒙特卡洛方法，其作用是在计算机中得到任意分布的一个样本（一般采样次数越多越准确）。核心思想是使用随机数（或更常见的伪随机数0-1均匀分布）来解决一些复杂的计算问题。
第二步，理解马尔科夫链以及矩阵的极限等概念，马氏链要收敛，但是采样多少次才会收敛，在工程实践中我们更多的靠经验和对数据的观察来指定 Gibbs Sampling 中的 burn-in 的迭代需要多少次。
[随机采样和随机模拟：吉布斯采样Gibbs Sampling](http://blog.csdn.net/pipisorry/article/details/51373090)
[MCMC](http://www.ctolib.com/topics-105669.html)

另外这个实现 https://github.com/elplatt/lda-gibbs-em 还包含了em迭代过程。
[Peacock LDA分布式实现](http://forum.ai100.com.cn/blog/thread/ml-2015-03-03-3816245109043572/)
[LDA工程实践之算法篇](http://www.flickering.cn/nlp/2014/07/lda%E5%B7%A5%E7%A8%8B%E5%AE%9E%E8%B7%B5%E4%B9%8B%E7%AE%97%E6%B3%95%E7%AF%87-1%E7%AE%97%E6%B3%95%E5%AE%9E%E7%8E%B0%E6%AD%A3%E7%A1%AE%E6%80%A7%E9%AA%8C%E8%AF%81/)

---

三种类型的主题模型：LSI，PLSI，LDA。LSI为$A_{m*n}=U_{m*k}*\Sigma_{k*k}*V_{k*n}$，其中A代表m个term、n篇文章，U代表m个term、k个topic，V代表k个topic、n篇文章。开源库[gensim](https://github.com/mahatmaWM/gensim)均有实现。

PLSI，LDA讲解的比较清楚的博文[通俗理解LDA主题模型](http://blog.csdn.net/v_july_v/article/details/41209515)，[LDA主题模型可视化分析](http://cpsievert.github.io/LDAvis/reviews/vis/#topic=1&lambda=0&term=)。

关于lda的评测[LDA主题模型的评估](http://blog.csdn.net/pipisorry/article/details/42460023)，了解Perplexity(混乱)与熵的定义，用熵来定义Perplexity，其值越小越好（其值越小说明对应的熵也越小，集合处在一种比较有序的状态，模型较好）。

---

em与gibbs的区别？？
EM算法的一般步骤为：

1. 随机选取或者根据先验知识初始化；
2. 不断迭代下述两步
①给出当前的参数估计，计算似然函数的下界
②重新估计参数θ，即求，使得
3. 上述第二步后，如果收敛（即收敛）则退出算法，否则继续回到第二步。

[概率语言模型及其变形系列](http://www.52nlp.cn/%E6%A6%82%E7%8E%87%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B%E5%8F%8A%E5%85%B6%E5%8F%98%E5%BD%A2%E7%B3%BB%E5%88%97-lda%E5%8F%8Agibbs-sampling)
