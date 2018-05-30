**目录**

[toc]

## MLND 

### 毕业开题报告
作者：RayZen
日期：2018 年 5 月 27 日

### 项目背景
`Rossmann` 是经营了超过 `3,000` 家遍布欧洲 `7` 个国家的公司。现在 `Rossmann` 经营者希望能够预测六周的每日销售情况，而它的销售情况收到多个因素影响，例如：促销、竞争对手、学校和国家节假日、季节性因素以及本地化等因素等的影响。因为根据每家店的管理者根据当地的实际情况进行了销售预测，预测结果的准确度差异太大${^{[1]}}$。

本次项目，`Rossmann` 的管理者希望能够通过提高预测的准确性来帮助他们更有效的做出工作计划，以及同时达到提高效率和机动性。对此，提出了需要增强预测的 **稳定性**。

机器学习，主要是通过相应的数据来学习确认一种现实世界的模式，或者不用显式编程和构建模型的方式来进行预测。典型的被分为监督式学习和非监督式学习${^{[2,3]}}$。其中监督式学习根据有“标签”的数据进行学习，通过得到的数据模型应用于通过输入数据获得结果（另外根据不同的输入数据或者特殊的反馈方式，又被分为**半监督式学习**、**增强式学习**等）；另外非监督式学习是通过无“标签”数据进行学习，以其找到数据中的结构信息以备其他学习方式使用。另外在实际应用场景下，监督式学习主要是回归和分类两种方式；而非监督式学习的应用包括聚类、推荐系统等方面。

### 问题描述
根据 `Rossmann` 经营者提出的要求，构建模型以提高预测接下来的六周准确的稳健性。也就是说需要对经营额进行预测，这是一个连续性数据；`data` 是具有 `label` 的数据${^{[1]}}$。

那么解决这个问题的方式，是从监督式学习中的回归方式来解决这个问题。因此此项目的重点是增强预测的稳定性，以到达可以对实际经营情况进行稳定预测的目的。

### 数据探索
本次项目中的文件列表如下：

* `train.csv`：具有 `label` 的历史销售数据，需要用于训练构建模型
* `test.csv`：无 `label` 的历史销售数据，需要用于测试训练的模型
* `sample_submission.csv`：提交的预测数据的正确格式样本文件
* `store.csv`：补充的 `Rossmann` 经营的 `1115` 家商店的信息

而在本次项目中需要的数据的 `features` 信息如下：


|Feature Name|Description|More Information|
|:-----------:|:---------|:---------------|
|Store|每家商店的独一编号|
|Sales|给定的每天的销售额|
|Customers|给定的每天的消费者数量|
|Open|指示说明商店是否营业|`0`=`closed`;`1`=`open`|
|StateHoliday|指示说明国家法定节假日|`a`=`public holiday`;`b`=`Easter holiday`;`c`=`Christmas`;`0`=`None`|
|SchoolHoliday|说明商店是否都受到学校关闭的影响|
|StoreType|说明商店的类型|`a`,`b`,`c`,`d`|
|Assortment|说明商店的差异经营策略的类型|`a`=`basic`,`b`=`extra`,`c`=`extended`|
|CompetitionDistance|最近竞争者的距离|
|CompetitionOpenSicne|最近竞争者的开始营业时期|`CompetitionOpenSicneMonth`=开始经营月份;`CompetitionOpenSicneYear`=开始经营年|
|Promo|说明给定日期时商店是否有进行促销|
|Promo2|说明商店是否有进行连续的促销活动|`0`=`store is not participating`;`1`=`store is participating`|
|Promo2Since|描述了开始参与连续性促销的日期|`Promo2SinceWeek`=参与促销的月份;`Promo2SinceYear`=参与促销的年份|
|PromoInterval|描述有连续性促销的间隔|哪些月份有连续性促销|

本数据集中总共可以使用的 `features` 数量为 `18`，而数据量为 `1,017,209` 条记录。同时从以上的 `features` 描述和应当使用的数据类型来看，某些数据是连续性数值数据，例如：`Sales`,`StoreType`,`Customers`,`CompetitionDistance`；离散型二分类数据，以 `0` 和 `1` 的数值来表达一些特殊性信息，例如：`Open`, `Promo`,`Promo2`,`SchoolHoliday`；离散型多分类数据，通过不同的字符表示不同信息，例如：`StateHoliday`,`StoreType`,`Assortment`；序列型数据，主要为序号以及日期类型数据；另外有一类是文本描述类数据，例如:`PromoInterval`。对现有 `features` 进行绘图，如下：

![](img/scatter_matrix.png)

针对以上不同类型的数据需要进行不同的处理，主要是分类型数据和文本描述型数据，这类数据需要转换为相应的可分析的数据语言。


### 解决方法描述



### 参考

1. [Rossmann Store Sales.](https://www.kaggle.com/c/rossmann-store-sales)
2. [Machine learning .](https://en.wikipedia.org/wiki/Machine_learning)
3. [Machine Learning for Humans.](https://medium.com/machine-learning-for-humans/why-machine-learning-matters-6164faf1df12)
