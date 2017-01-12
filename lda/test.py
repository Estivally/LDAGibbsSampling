# coding=utf-8

import os
import lda

# 读入训练文章 ['cat duck', 'cat dog']
foldpath = '/Users/wangming/workspace/LDAGibbsSampling/data/'
docs = [' '.join(open(foldpath + f, 'r').readlines()) for f in os.listdir(foldpath)]

# 切词
tokenized_docs = lda.tokenize(docs, min_times=3, max_ratio=1.0, min_word_size=4)

lda = lda.LDASampler(
    docs=tokenized_docs,
    num_topics=50,
    alpha=0.25,
    beta=0.25)

print 'topic assignments for each of 10 iterations of sampling:'
for i in range(1000):
    if i % 100 == 0:
        print 'iter', i
    zs = lda.assignments
    # print '[%i %i] [%i %i]' % (zs[0][3], zs[1][3], zs[2][3], zs[3][3])
    lda.next()

print 'words ordered by probability for each topic:'
tks = lda.topic_keys()
for i, tk in enumerate(tks):
    print i, tk
exit(0)
print 'document keys:'
dks = lda.doc_keys()
for doc, dk in zip(docs, dks):
    print doc, dk

print 'topic assigned to each word of first document in the final iteration:'
lda.doc_detail(0)
