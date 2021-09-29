# 古词生成器
基于Google BERT模型的古词生成器，使用[simplebert](https://github.com/gaolichen/simplebert)框架编写。

## 主要功能

- 使用[中华古诗词数据库](https://github.com/chinese-poetry/chinese-poetry)的宋词进行训练
- 给定词牌名，随机生成一首词

随机生成词--浣溪沙：
```
浣溪沙
小院朱门春已深。绿阴如幄日如薰。
一帘风雨雨如丝。玉笋轻笼红粉面，
金炉香暖紫霞腮。不禁春梦苦相催。
```

## 运行方法
- 安装simplebert： `pip install simplerbert`
- 在本目录的上一层目录下，运行`python -m cigenerator.train`
