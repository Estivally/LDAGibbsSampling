# 备忘

本代码的实现参考[论文](http://www.pnas.org/content/101/suppl_1/5228.full.pdf)或者[LDA工程实践实现正确性验证 1、2](http://www.flickering.cn/nlp/2014/07/lda%E5%B7%A5%E7%A8%8B%E5%AE%9E%E8%B7%B5%E4%B9%8B%E7%AE%97%E6%B3%95%E7%AF%87-1%E7%AE%97%E6%B3%95%E5%AE%9E%E7%8E%B0%E6%AD%A3%E7%A1%AE%E6%80%A7%E9%AA%8C%E8%AF%81/)这里总结的公式。
除了本文的实现，[另一个实现](https://github.com/elplatt/lda-gibbs-em)还包含了em迭代求参。

doc目录中八卦那篇文章对lda所涉及到的数学知识做了一个总结（以及这个pdf的网页版本[LDA-math-文本建模](http://cos.name/2013/03/lda-math-text-modeling/)），另外一篇讲gibbs采样的分析也很详细。

另外关于lda的评测，了解Perplexity(混乱)与熵的定义，用熵来定义Perplexity，其值越小说明对应的熵也越小，集合处在一种比较有序的状态，模型较好。具体说明见
[LDA工程实践实现正确性验证 1、2](http://www.flickering.cn/nlp/2014/07/lda%E5%B7%A5%E7%A8%8B%E5%AE%9E%E8%B7%B5%E4%B9%8B%E7%AE%97%E6%B3%95%E7%AF%87-1%E7%AE%97%E6%B3%95%E5%AE%9E%E7%8E%B0%E6%AD%A3%E7%A1%AE%E6%80%A7%E9%AA%8C%E8%AF%81/)**总结的非常好**。
还有分布式的实现 [Peacock LDA分布式实现](http://forum.ai100.com.cn/blog/thread/ml-2015-03-03-3816245109043572/)

---

三种类型的主题模型：LSI，PLSI，LDA。LSI为$A_{m*n}=U_{m*k}*\Sigma_{k*k}*V_{k*n}$，其中A代表m个term、n篇文章，U代表m个term、k个topic，V代表k个topic、n篇文章。开源库[gensim](https://github.com/mahatmaWM/gensim)均有实现。PLSI，LDA讲解的比较清楚的博文[通俗理解LDA主题模型](http://blog.csdn.net/v_july_v/article/details/41209515)。

[概率语言模型及其变形系列](http://www.52nlp.cn/%E6%A6%82%E7%8E%87%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B%E5%8F%8A%E5%85%B6%E5%8F%98%E5%BD%A2%E7%B3%BB%E5%88%97-lda%E5%8F%8Agibbs-sampling)
