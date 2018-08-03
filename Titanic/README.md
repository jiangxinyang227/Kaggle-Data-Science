### 来自Kaggle竞赛——Titanic幸存者预测
***
#### 数据集 Data
* train.csv 训练数据集
* test.csv 测试数据集
#### 特征
* passengerID ID号
* pclass 社会等级
* sex 性别
* Age 年龄
* sibsp 家庭成员关系，同辈（例如丈夫，表哥，堂姐）
* parch 家庭成员关系，上下辈（例如母亲，女儿）
* ticket 船票信息
* fare 票价
* cabin 房间号
* embarked 登船口岸
#### 预测特征
* survived，二分类问题，0：死亡；1：幸存
#### 模型
* 采用了stacking集成算法
#### 最终排名
    1208/10390