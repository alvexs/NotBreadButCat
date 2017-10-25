# NOT BREAD BUT A CAT

Lab3 on course "Decision Support Methods"

Лабараторная работа №3 по курсу "Методы поддержки принятия решений"


[Report](https://github.com/alvexs/NotBreadButCat/blob/master/Lab2_report.pdf)


## Usage:

### Own model

```
fab demo
fab train:<lr>,<image_dir> (default - 'fab train' or 'fab train:0.001,data/trainbread')
fab predict:<file>
```

### Fine-tuned VGG16 model

```
fab demo2
fab train2:<lr>,<image_dir> (default - 'fab train' or 'fab train:0.001,data/trainbread')
fab predict2:<file>
