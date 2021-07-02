# WeChat-Big-Data-Challenge-2021
2021年微信大数据挑战赛-NN Baseline，比赛详细说明在：https://algo.weixin.qq.com/intro。



### 1. 总结

------

该方案是基于DNN的baseline方案，线上加权uauc=0.667139。若想进一步提升指标，可以考虑如下几方面：

- 文本特征：如何利用数据集中提供的文本特征，如description、ocr以及asr（分别有word和char为单位的）。尝试过多种方案，如TF-IDF（如何解决冷门word或char，比如只出现一次）获取每段文本最重要的几个词；word2vector获取word embedding，然后通过pooling的方式形成句子的embedding。这些方法在实验中均未取得效果。建议可以考虑通过textcnn或者lstm的方式，学习word embedding。
- NN模型的优化：本方案仅采用了三层DNN，可以考虑换成deepfm或者xdeepfm，应该能够获得一定的提升。
- 模型融合：
  - 从各位选手的分享来看，若想取得较好的成绩，融入LihgtGBM模型是一个较好的方式。本方案使用全量数据五折交叉进行训练，因此除了使用视频时长外其他没有使用连续特征（数据穿越）。
  - 后期模型融合除了设置不同的随机种子，可以考虑不同特征、不同深度模型等方式。
- 训练方式
  - 全量单折 VS 全量多折：全量多种的方式离线指标波动小，对于线上指标的对齐也较好；但是全量单折的方式可以融合连续特征、userid embedding以及authorid embedding等。







### 2.环境配置

------









### 3.运行配置

------









### 4.目录结构

------







### 5.运行流程

------





### 6.模型及特征

------







### 7.模型结构

------





### 8.相关文献

------

