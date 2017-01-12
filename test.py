import lda

import os


foldpath = '/Users/wangming/workspace/temp/LDAGibbsSampling/data/LdaOriginalDocs/'
docs = [' '.join(open(foldpath + f, 'r').readlines()) for f in os.listdir(foldpath)]

#docs = ['cat cat sabdf sdfdf', 'cat dog']
#print 'docs:', docs

tokenized_docs = lda.tokenize(docs, min_times=3, max_ratio=1.0, min_word_size=4)

# print 'tokenized docs:', len(tokenized_docs)
# for item in tokenized_docs:
#     print len(item)

# exit(0)
lda = lda.LDASampler(
    docs=tokenized_docs,
    num_topics=5,
    alpha=0.25,
    beta=0.25)
# exit(0)
print 'topic assignments for each of 10 iterations of sampling:'
for _ in range(1000):
    zs = lda.assignments
    # print '[%i %i] [%i %i]' % (zs[0][3], zs[1][3], zs[2][3], zs[3][3])
    lda.next()
print
# exit(0)
print 'words ordered by probability for each topic:'
tks = lda.topic_keys()
for i, tk in enumerate(tks):
    print i, tk
print

print 'document keys:'
dks = lda.doc_keys()
for doc, dk in zip(docs, dks):
    print doc, dk
print

print 'topic assigned to each word of first document in the final iteration:'
lda.doc_detail(0)
