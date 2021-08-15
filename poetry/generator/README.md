# 古诗词生成器
基于Google BERT模型的古诗词生成器，使用[simplebert](https://github.com/gaolichen/simplebert)框架编写。

## 主要功能

- 使用[中华古诗词数据库](https://github.com/chinese-poetry/chinese-poetry)的唐诗和宋词进行训练
- 可随机生成一首古诗
- 可续写一首古诗
- 可写一首藏头诗

随机生成古诗一：
```
夕阳西北望，山色与江平。
野旷风烟静，川长日月明。
孤城连海角，远树出江城。
何处无人到，秋潮自不鸣。
```
随机生成古诗二：
```
四海无双老杜家，胸中自有一丘沙。
诗中有句元无物，不是诗人是酒家。
```
续写`万水千山总是情`：
```
万水千山总是情，一轮明月一钩明。
莫嫌夜半风吹起，月下无人月自明。
```

藏头诗`好好学习`
```
好山不在山林外，
好竹长留竹院中。
学到古人成老境，
习将今日作春风。
```

## 运行方法
- 安装simplebert： `pip install simplerbert`
- 在本目录的上一层目录下，运行`python -m peotry_generator.train`

