# 备忘

Gibbs实现参考原始论文[Parameter estimation for text analysis](http://www.arbylon.net/publications/text-est.pdf)

另外一些值得看的博文：[LDA工程实践实现正确性验证 1、2](http://www.flickering.cn/nlp/2014/07/lda%E5%B7%A5%E7%A8%8B%E5%AE%9E%E8%B7%B5%E4%B9%8B%E7%AE%97%E6%B3%95%E7%AF%87-1%E7%AE%97%E6%B3%95%E5%AE%9E%E7%8E%B0%E6%AD%A3%E7%A1%AE%E6%80%A7%E9%AA%8C%E8%AF%81/) 
[另一个实现](https://github.com/elplatt/lda-gibbs-em)还包含了变分EM迭代求参。

doc目录中八卦那篇文章对lda所涉及到的数学知识做了一个总结（以及这个pdf的网页版本[LDA-math-文本建模](http://cos.name/2013/03/lda-math-text-modeling/)

还有分布式的实现 [Peacock LDA分布式实现](http://forum.ai100.com.cn/blog/thread/ml-2015-03-03-3816245109043572/)

---

三种类型的主题模型：LSI，PLSI，LDA。LSI为$A_{m*n}=U_{m*k}*\Sigma_{k*k}*V_{k*n}$，其中A代表m个term、n篇文章，U代表m个term、k个topic，V代表k个topic、n篇文章。
开源库[gensim](https://github.com/mahatmaWM/gensim)均有实现。
PLSI，LDA讲解的比较清楚的博文[通俗理解LDA主题模型](http://blog.csdn.net/v_july_v/article/details/41209515)