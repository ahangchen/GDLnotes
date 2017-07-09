# TensorFlow 安装踩坑日志
## Install TensorFlow

安装教程就在TensorFlow的官网上>>>[点击查看](https://www.tensorflow.org/install/)

按照官方的流程装就好了，这里讲一下几种方式的特点：

1. pip: 安装在全局的python解释器中，简单

2. Third party: Virtualenv, Anaconda and Docker：都能创建tensorflow独立的编译环境，但就是多了一份包

3. Source: 能够适应不同的python版本（比如编译一个3.5版的），但源码编译可能有许多坑

- ubuntu安装时，需要注意自己的python - pip - tensorflow版本是否对应（比如是否都是2.7），
- 使用sudo命令时，注意自己的环境变量是否变化（会导致pip或python命令对应的版本变化）
- 具体讲一下ubuntu安装tensorflow流程：
  - 安装anaconda2
  - 确定自己终端的pip和python版本：
  ```
    $ pip -V && python -V
  ```
    确认使用的是否都来自anaconda，如果不是，则应该使用类似这样的命令运行对应的pip：
  ```
    $ /home/cwh/.conda/envs/tensorflow/bin/pip -V
  ```
  
  即最好安装到tensorflow自己的python环境里，不跟anaconda原来的环境混淆
  
    使用sudo命令时最好也看一下版本
    
  - 使用anaconda创建一个tensorflow虚拟环境：
  ```
    $ conda create -n tensorflow python=2.7
  ```
  - 切换到tensorflow环境下（实际上是更换了环境变量里的pip和python），下载安装tensorflow，需要sudo权限
  ```
    $ source activate tensorflow
    (tensorflow)$ sudo pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.8.0rc0-cp27-none-linux_x86_64.whl
    $ source deactivate
  ```
  注意如果安装的是gpu版本，还需要按照官网说明安装cuda和cudaCNN，具体教程看这个[视频](https://www.youtube.com/watch?v=cVWVRA8XXxs)，不能科学上网的访问这个[地址](http://www.tudou.com/programs/view/MEnGrbSTui8/?bid=03&pid=02&resourceId=391713117_03_0_02)，注意一下[你的显卡算力](https://developer.nvidia.com/cuda-gpus)
  - 如果pip安装速度慢，不要换pip源，复制whl名字，去谷歌一搜，找到对应的whl下下来，然后pip install xxx.whl，整个过程比全pip安装要快得多
  - 如果setuptools安装失败，报”Cannot remove entries from nonexistent file”，就要用
  ```shell
  $ pip install --ignore-install setuptools
  ```
    覆盖安装

  - 安装成功后就可以在tensorflow的python环境下，执行import tensorflow看看了。
