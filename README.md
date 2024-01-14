在命令行输入python main.py --m <model_path> --s <separate>运行。

其中，<model_path>是保存的模型名，<separate>代表是否要分离测试，0代表否，1代表是。

其他可选参数：
--d 训练数据路径，默认为./data
--e 模型嵌入维度
--h 模型注意力头数目
--l 模型层数
--r dropout率
--ls 末尾线性层的神经元密度
--b 批次大小

输出的模型文件保存在./models目录下，学习曲线保存在./plots目录下。

运行环境：除了基础的python库和Pytorch库，还需要安装torch_geometric

