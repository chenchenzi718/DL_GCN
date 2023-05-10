cora数据集包含了机器学习的文章，这些文章可以被分类为以下七个类：

		Case_Based
		Genetic_Algorithms
		Neural_Networks
		Probabilistic_Methods
		Reinforcement_Learning
		Rule_Learning
		Theory

论文的选择方式是，在最后的语料库中，每篇论文引用或被至少一篇其他论文引用。整个语料库共有2708篇论文。
在词干和删除词缀后，我们留下了1433个独特单词的词汇表。删除所有文档频率低于10的单词。

整个文档包含以下两个文件：
1. .content文件包含了如下内容：<paper_id> <word_attributes>+ <class_label>
<paper_id>是每篇文章的唯一的标识；<word_attributes>用0表示某个词不在文章中，1表示在文章中；<class_label>给定了文章类别
2. .cites文件包含了一张语料库中的文献引用图，每一行用<ID of cited paper> <ID of citing paper>表示
前者是被引用文章，后者是引用文章