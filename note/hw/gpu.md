# 浪潮 NF5280M3 GPU选型

## 必备条件
- 支持Tensorflow（即算力超过3的NVIDIA显卡）

> NVIDIA显卡算力：[link](https://developer.nvidia.com/cuda-gpus)

- 可虚拟化：因为我们需要在浪潮服务器上虚拟化多台ubuntu做分布式运算，所以需要虚拟化

NVIDIA与VMWare关于GPU虚拟化的[成果](http://www.nvidia.cn/object/grid-boards-cn.html)

初步锁定GRID K1或GRID K2

##  NF5280M3是否支持NVIDIA显卡？
- 支持：[13年一次比赛](http://scc.ustc.edu.cn/yjdt/201305/t20130506_150923.html)就是用的这两家的产品
- 使用的是NVIDIA的Tesla K20GPU加速卡
- Tesla系列显卡算力可以在上面的显卡算力中查到，

## 存疑
- Grid显卡是否支持tensorflow/cuda？

> 支持：[CUDA FAQ](https://developer.nvidia.com/cuda-faq)提及:CUDA is a standard feature in all NVIDIA GeForce, Quadro, and Tesla GPUs as well as NVIDIA GRID solutions.

- Grid显卡和Tesla显卡是不同系列的吧，那么Grid显卡是否能够安装到NF5280M3上？
- Grid显卡在算力表中没有，是否达到算力3？
