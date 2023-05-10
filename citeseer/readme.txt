此目录包含Citeeer数据集的选择。


这些论文分为以下六类：
			Agents
			AI
			DB
			IR
			ML
			HCI

论文的选择方式是，在最后的语料库中，每篇论文引用或被至少一篇其他论文引用。整个语料库共有3312篇论文。

在词干和删除词缀后，我们留下了3703个独特单词的词汇表。删除所有文档频率低于10的单词。

该目录包含两个文件：
1. .content文件包含了如下内容：<paper_id> <word_attributes>+ <class_label>
<paper_id>是每篇文章的唯一的标识；<word_attributes>用0表示某个词不在文章中，1表示在文章中；<class_label>给定了文章类别
2. .cites文件包含了一张语料库中的文献引用图，每一行用<ID of cited paper> <ID of citing paper>表示
前者是被引用文章，后者是引用文章
注： 与cora不同的是它这里的ID不一定是数字