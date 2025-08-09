### Megatron入门资料

https://zhuanlan.zhihu.com/p/629121480

### 环境

```shell
eval "$(/nas/njw1123/miniconda3/bin/conda shell.bash hook)"
conda activate megatron
```

### 启动训练

```shell
cd Megatron-LM/examples/video_fun/
bash train_only_dit.sh ## 参数含义可以参考megatron/training/arguments.py
```

部分参数说明(可能需要修改的参数)：

```shell
NODE_RANK # 当前节点的rank
MASTER_ADDR  # 主节点IP
NUM_NODES # 节点数量

--tensor-model-parallel-size  # tp大小
--pretrained_model_path # 模型路径
--transformer3d_config_path # 模型配置文件路径

```

### 模型转换

转换脚本路径：

``` shell
Megatron-LM/examples/convert_WanDiT.sh
Megatron-LM/examples/convert_WanDiT.py
```

需要修改convert_WanDiT.py中的参数设置模型读取和存储路径：

```python
dit_path = "path/to/Wan2.1-Fun-V1.1-14B-InP" # 原模型checkpoint的位置
save_path = "path/to/dit" # megatron 格式的checkpoint的存储位置
```

### Megatron 代码迁移与实现解析

- 这部分主要介绍修改了megatron哪些部分的实现，直接拷贝原仓库实现的部分会进行省略，以及部分训练流程中的环节并没有修改，可以参考网上的Megatron源码解读理解

- 训练的起始文件是pretrain_only_dit.py
  - patchify_target函数将target进行patchify，然后从序列维度截取当前设备需要处理的片段(sp(tp)和cp并行需要在序列维度切分input和label)
  - modify_transformer3d_config函数会读取huggingface上下载的wan模型的config，并将对应的配置如num_layers, hidden_size等进行覆盖，因此pretrain_only_dit.sh中设置的部分参数只起占位的作用的，因为megatron要求必须设置这些参数。
  - model_provider根据配置定义模型
    - WanTransformer3DModel模型的定义在megatron/core/models/VideoX_Fun
    - 在定义WanTransformer3DModel时需要提前确定模型中每层layer的结构，代码在megatron/core/models/VideoX_Fun/transformer3d_layer_specs.py中，类似于搭积木，需要指定layernorm，linear，core_attention等分别使用什么实现，具体选择的layernorm，linear等的实现，要么时transformer engine库中提供，要么是自定义实现的，主要位于megatron/core/transformer文件夹下
  - get_batch函数迭代data_iterator获取下一个batch的数据
  - loss_func函数对loss进行计算，其实model的output已经是loss了，这里在megatron之前的视线中会进行loss_mask等操作，但是现在的实现并没有添加padding tokens，不需要mask
  - forward_step函数定义了从获取一个batch的数据到执行完一次前向计算过程，加噪等输入数据等处理都在这里，保证了分布式情况下采样是正确的，目前的输入是随机的，后续需要替换成预处理的数据输入。
  - train_valid_test_datasets_provider函数进行数据集的定义，目前还未确定预处理数据集的组织形式，所以使用了fake的数据集占位。

