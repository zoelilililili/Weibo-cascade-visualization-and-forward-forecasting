# 多方法融合的微博信息级联传播研究：可视化、宏观与微观预测系统

## 项目简介

社交媒体中的信息级联传播是网络科学与计算社会科学的重要研究课题。本项目以微博平台为例，构建了一个集可视化分析、宏观趋势预测与微观机制建模于一体的多方法融合研究框架。通过综合运用机器学习与深度学习技术，系统性地探索微博信息传播的模式与规律，为舆情监测、内容推荐和传播干预提供科学依据。

核心亮点：
- 🔍 多维度可视化分析：展示信息传播的拓扑结构、时空演化与深度分布
- 📈 宏观趋势预测：基于早期传播特征预测最终转发规模
- 🎯 微观行为建模：结合社交关系预测个体用户的转发决策
- 🔗 完整研究闭环：从现象观察到机制建模再到预测应用

## 作者
李哲伟，复旦大学大数据学院，23307130111@m.fudan.edu.cn 
章一诺，复旦大学大数据学院，23307110124@m.fudan.edu.cn 
王卓熙，复旦大学大数据学院，23300130025@m.fudan.edu.cn


## 目录
1. [研究背景]
2. [项目结构]
3. [数据说明]
4. [快速开始]
5. [研究方法与成果]
6. [致谢与联系]

## 1. 研究背景

随着社交媒体的快速发展，信息级联传播已成为影响社会舆情、商业营销和公共安全的关键现象。微博作为中国最具影响力的社交媒体平台之一，其信息传播机制的研究具有重要的理论和实践意义。

正如 Antoine de Saint-Exupéry 在《小王子》中所说：“人们只有用心去看，才能看到真实。事物的真实本质是肉眼无法看到的。”本项目正是希望通过多方法融合的研究视角，揭示微博信息传播背后的复杂机制。

本项目探索社交媒体环境下信息传播的新模式，能够为舆情监控、内容推荐提供技术支持，并帮助理解和管理网络信息传播。

本项目采用"可视化探索-宏观建模-微观分析"的三层研究框架：可视化——直观展示传播现象，宏观——预测整体传播趋势，微观——理解个体传播行为

## 2. 项目结构

```
微博信息级联传播研究/
├── data/                                 # 数据目录
│   ├── macro_data/                       # 宏观预测数据
│   ├── micro_data/                       # 微观预测数据
│   │   ├── cascades                      # 转发序列
│   │   ├── cascades_retweet_trees        # 转发树
│   │   └── edges                         # 社交网络图
├── data_analysis/                        # 数据可视化脚本目录
│   ├── visualize_retweet_sequence.py     # 转发量-时间可视化
│   ├── plot_retweet_curves.py            # 转发量、热度、地区可视化
│   ├── visualize_depth.py                # 深度相关可视化
│   ├── sunburst.py                       # 传播结构旭日图
│   ├── retweet_trees_for_visualization1/ # 存储需要转发、热度等可视化的数据
│   ├── retweet_trees_for_visualization2/ # 存储需要深度相关可视化的数据
│   └── outputs/                          # 存储可视化结果图
├── models/                               # 宏观预测和微观预测模型
│   ├── macro_predictor/                  # 宏观预测模型
│   │   ├── outputs/                      # 模型输出
│   │   ├── XGBoost_predictor.py          # 主预测程序
│   ├── micro_predictor/                  # 微观预测模型
│   │   ├── cascade_prediction.py         # 转发树可视化
│   │   └── convert_trees_to_casades.py   # 时间序列分析
├── WeiboSpider/                          # 爬虫代码模块
│   ├── results/                          # 爬取日志
│   ├── retweet_tree_crawler.py           # 爬虫脚本
│   ├── user_crawler.py                   # 爬取社交网络脚本
│   ├── cookie.txt                        # cookie
│   └── mids.txt                          # 爬取微博mid集合
├── report.pdf                            # 项目报告
├── requirements.txt                      # Python依赖包
└── README.md                             # 本文件
```

## 3. 数据说明

数据来源于新浪微博平台，通过自定义爬虫采集完整的微博转发树结构。

## 4. 快速开始

### 4.1 环境要求

```bash
pip install requirements.txt
```
执行上述命令即可配置好环境，若下载速度慢可以替换成国内镜像源。


### 4.2 数据爬取

#### 4.2.1 cookie获取

登录新浪微博，进入主页，按住键盘上Fn+F12，进入开发者模式，选择Fetch/XHR，刷新网页，点击名称中第二行*config，点开后，可以看见标头中有Cookie字样，复制Cookie后的字样到本项目的cookie.txt文件中。

注意Cookie具有时效性。

#### 4.2.2 微博mid获取

登录新浪微博，点击需要爬取的微博进入其微博主页，按住键盘上Fn+F12，进入开发者模式，选择Fetch/XHR，刷新网页，其中带有“id=”或“mid=”后面的16位数字即为此微博的mid。获取需要爬取的微博的mid，按行写入mids.txt中。

#### 4.2.3 爬取微博转发树

```bash
cd WeiboSpider
python retweet_tree_crawler.py
```
在完成4.2.1和4.2.2步骤后，在终端里执行上述命令，即可运行微博转发树爬虫脚本。
若需要修改爬虫参数，在retweet_tree_crawler.py中配置部分修改相应参数即可。
爬取的转发树以JSON文件保存于于data\raw中，此次爬虫日志保存在\WeiboSpider\results中。

数据筛选：爬取到的微博转发树可能质量不高，可以通过可视化来筛选数据：
```bash
cd data_analysis
python visualize_retweet_sequence.py
```
之后此脚本输出一个html文件，可以看到爬取的所有转发树前TIME_LIMIT（脚本中可以调节）秒内的转发量随时间变化曲线，以此观察转发情况，不符合预期（如前两个小时数据量过少）等可以删除此条数据。
#### 4.2.4 爬取社交网络

对于已经获得的的转发树，如果需要获取相应的关注关系构成的社交网络，先将需要的转发树复制到存储在cascades_retweet_trees，之后运行以下命令
```bash
cd WeiboSpider
python user_crawler.py
```
这将每个转发树对应的用户关注关系以JSON格式存储于data\micro_data\edges中。

### 4.3 信息级联可视化

#### 4.3.1 转发量、热度、地区-时间曲线
将需要可视化的转发树JSON文件复制于data_analysis\retweet_trees_for_visualization1中。
运行以下代码
```bash
cd data_analysis
python plot_retweet_curves.py
```
可以得到转发量、热度、总地区数量随时间变化曲线。

#### 4.3.2 深度可视化及传播结构旭日图
将需要可视化的转发树JSON文件复制于data_analysis\retweet_trees_for_visualization2中。
运行以下代码
```bash
cd data_analysis
python sunburst.py
```
可以得到转发结构旭日图。
运行以下代码
```bash
cd data_analysis
python visulize_depth.py
```
可以得到深度信息图。


### 4.4 宏观预测

在爬取到一定数量的转发树后（本项目已有一百条左右数据），运行以下命令
```
cd models
cd macro_predictor
python XGBoost_predict.py
```
此脚本输出训练中预测结果和实际结果的散点图。

### 4.5 微观预测

先提取出转发树中需要的用户序列信息，并且按时间顺序排列为列表，运行以下命令：
```
cd models
cd micro_predictor
python convert_trees_to_cascades.py
```
提取后的信息以JSON格式储存于data\micro_data\cascades中，之后再运行以下命令：
```
python cascades_prediction.py
```
此温江利用data\micro_data\cascades和data\micro_data\edges中的数据训练模型并输出预测结果评估指标。

## 5. 研究方法与成果

见本项目报告。

## 6. 致谢与联系

### 6.1 致谢
我们衷心感谢所有为本项目做出贡献的研究者和开发者。每一个建议、每一次讨论、每一行代码都是项目成长的重要养分。

特别感谢：
- 亲爱的老己，在忙碌的期末阶段，为本项目做出了巨大的贡献。
- 《社交网络挖掘》课程的助教许婧函学姐和阳德青老师，他们的指导和建议为本项目提供了不少帮助。
- Chatgpt和Deepseek，它们聪慧的大脑与优秀的执行了为我们代码编写省去了大量时间。

正所谓：“众人拾柴火焰高。”我们相信，只要大家的热情在，一起努力总会取得很好的结果。

### 6.2 联系信息
如果您有任何问题、建议或合作意向，欢迎通过以下方式联系我们：

李哲伟，复旦大学大数据学院，23307130111@m.fudan.edu.cn 
章一诺，复旦大学大数据学院，23307110124@m.fudan.edu.cn 
王卓熙，复旦大学大数据学院，23300130025@m.fudan.edu.cn


> "程序是为人类读写的，不是为机器执行的。" —— Donald Knuth, 《计算机程序设计艺术》

感谢您对本项目的关注！我们期待与您一起探索社交媒体信息传播的奥秘。
