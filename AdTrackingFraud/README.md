### 来自Kaggle竞赛——app点击下载预测
该问题属于二分类问题，预测点击后下载的概率。
#### 数据集
* train.csv 训练数据集，数据量非常大
* train_sample.scv 训练数据集样本
* test.csv 测试数据集
* test_supplement.csv 测试数据集的补充
#### 特征
* ip 设备的ip地址
* app 被下载的app的名称
* device 设备类型（如iphone6、iPhone7）
* os 设备的操作系统
* channel 设备来源渠道
* click_time 点击广告的时间
* attributed_time 下载app的时间（此特征毫无意义）
* is_attributed 预测的特征，是否已下载
#### 模型
采用微软的lightgbm模型