# 用CNN做猫狗图片分类

## 数据：
- 使用[kaggle](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data)的猫狗分类的训练和测试数据
- 所有图片转化成150*150大小

## 主要模型：
- 自定义的CNN模型
- 预训练的模型：VGG16,VGG19,ResNet50V2等

## 运行方法
- 在colab上打开notebook [dog_vs_cat.ipynb](dog_vs_cat.ipynb)
- 点击notebook左边的`Files`按钮，然后点`Upload to storage sesstions`，将`datasets.py`, `models.py`, `train.py`上传到colab虚拟机上
- 切换到GPU加速器，运行notebook

## 结果
