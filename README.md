## 甜橙金融初赛Rank1，复赛Rank16，复赛皮的嘛，就不谈了 ...
比赛链接：

http://www.dcjingsai.com/common/cmpt/2018%E5%B9%B4%E7%94%9C%E6%A9%99%E9%87%91%E8%9E%8D%E6%9D%AF%E5%A4%A7%E6%95%B0%E6%8D%AE%E5%BB%BA%E6%A8%A1%E5%A4%A7%E8%B5%9B_%E7%AB%9E%E8%B5%9B%E4%BF%A1%E6%81%AF.html

## 队伍名称：**Keep Real**

Ps: 应该改名为Keep Overfiting

## 解决方案：
我们初赛的方案就是通过追踪时间、设备、ip和经纬度等属性的变化来建模判断UID是否为黑产链，没啥复杂的操作，事实证明，这个方案不够稳健，所以复赛翻水水了。

复赛可通过删除分布差异较大的特征以及对使用的特征Rank化来一定程度上解决分布问题。

话说，我万万没想到transaction文件毒性这么大。

## 代码说明：
- gen_stat_feat.py      统计特征
- gen_w2v_feat.py       word2vec特征
- lgb_train.py  lgb训练模型


两份特征建模加权8:2比例融合即可0.792+，单独统计特征加UID列建模即可0.795。

有些赛友私信我说找不到代码里的文件，其实我是改名了，test_tag_r1.csv就是需要提交的测试集UID文件。
