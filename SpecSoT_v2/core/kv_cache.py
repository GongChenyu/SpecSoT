import torch
from typing import List


class KVCache:
    """
    A key-value cache for the model.

    This class provides a mechanism to maintain a growing cache of keys and values,
    particularly useful for models that benefit from caching previous states,
    like transformers during autoregressive decoding.

    Attributes:
        data (torch.Tensor): The tensor storing keys and values.
        current_length (int): Current length of the data being stored.
    """

    def __init__(self, data, current_length):
        """
        Initialize the KVCache.

        Args:
            data (torch.Tensor): Initial tensor to store the keys and values.
            current_length (int): Initial length of the data.
        """
        self.data = data
        self.current_length = current_length

    @property
    def shape(self):
        """Return the shape of the data tensor with updated length."""
        return (
            self.data.shape[0],
            self.data.shape[1],
            self.current_length.item(),
            self.data.shape[3],
        )

    def copy(self, indices: torch.Tensor, prev_length: int, dim: int = 2):
        """
        Copy values from the current data at specified indices to a new location.

        Args:
            indices (torch.Tensor): Indices of the data tensor to be copied.
            prev_length (int): Previous length before adding new data.
            dim (int, optional): Dimension along which copying should be performed. Default is 2.
        """
        tgt = self.data.index_select(dim, indices)
        dst = self.data.narrow(dim, prev_length, tgt.shape[dim])
        dst.copy_(tgt, non_blocking=True)
        self.current_length.fill_(prev_length + tgt.shape[dim])

    def restore_from_tensor(self, tensor: torch.Tensor, dim: int = 2):
        """
        从普通 tensor 恢复 KV cache 数据
        
        将 tensor 的数据复制到 cache 的开头，并设置正确的长度。
        常用于恢复之前保存的 cache 状态。
        
        Args:
            tensor (torch.Tensor): 要恢复的数据
            dim (int, optional): 数据维度，默认为 2 (seq_len 维度)
        """
        length = tensor.shape[dim]
        dst = self.data.narrow(dim, 0, length)
        dst.copy_(tensor, non_blocking=True)
        self.current_length.fill_(length)

    def cat(self, tensor: torch.Tensor, dim: int = 2):
        """
        Concatenate the given tensor with the current data.

        Args:
            tensor (torch.Tensor): The tensor to be concatenated.
            dim (int, optional): The dimension along which concatenation should be done. Default is 2.

        Returns:
            torch.Tensor: The data tensor after concatenation up to the current length.
        """
        dst = self.data.narrow(dim, self.current_length, tensor.shape[dim])
        dst.copy_(tensor)
        self.current_length.add_(tensor.shape[dim])
        return torch.narrow(self.data, 2, 0, self.current_length)

    def truncate(self, length: int):
        """
        裁剪到指定长度（只更新 current_length，不释放内存）

        用于 Draft Tree 生成后恢复 cache 状态，丢弃临时的 grow_tree cache。

        Args:
            length (int): 目标长度
        """
        self.current_length.fill_(length)

    def get_valid(self, dim: int = 2) -> torch.Tensor:
        """
        获取有效部分的视图（无复制）

        返回当前 cache 中有效数据的视图，不进行内存复制。

        Args:
            dim (int, optional): 数据维度，默认为 2 (seq_len 维度)

        Returns:
            torch.Tensor: 有效数据的视图
        """
        return self.data.narrow(dim, 0, self.current_length.item())


def initialize_past_key_values(model, max_length=2200, batch_size=1):
    """
    Initialize past key and value states for a given transformer model.

    This function prepares key-value cache structures for the model, allowing it to store and reuse
    past key and value states during autoregressive decoding, which can improve efficiency.

    Args:
        model (nn.Module): The transformer model for which past key-value states need to be initialized.
        max_length (int): Maximum sequence length for the cache.
        batch_size (int): Batch size for the cache (default: 1).

    Returns:
        tuple:
            - past_key_values (list): A list of KVCache objects for each layer in the model.
            - past_key_values_data (torch.Tensor): The tensor that will store all keys and values.
            - current_length_data (torch.Tensor): A tensor tracking the current length of keys/values in the cache.
    """
    # Extracting configuration from the model
    config = model.config
    # Initializing a tensor to store past keys and values for all layers

    devices=[]
    for i in range(config.num_hidden_layers):
        try:
            device = model.model.layers[i].self_attn.q_proj.weight.device
        except:
            device=model.layers[i].self_attn.q_proj.weight.device
        devices.append(device)
    past_key_values_data_list=[]
    startnum=0
    startdevice=devices[0]
    for id,i in enumerate(devices):
        if startdevice!=i:
            past_key_values_data = torch.zeros(
                startnum * 2,
                batch_size,
                config.num_key_value_heads,
                max_length,
                getattr(config, "head_dim", config.hidden_size // config.num_attention_heads),
                device=startdevice,
                dtype=model.dtype,
            )
            past_key_values_data_list.append(past_key_values_data)
            startdevice = i
            startnum=0
        startnum += 1
    past_key_values_data = torch.zeros(
        startnum * 2,
        batch_size,
        config.num_key_value_heads,
        max_length,
        getattr(config, "head_dim", config.hidden_size // config.num_attention_heads),
        device=startdevice,
        dtype=model.dtype,
    )
    past_key_values_data_list.append(past_key_values_data)
    # Initialize tensor to store the current length of the cached data for all layers.
    # [IMPORTANT] It needs to be kept on CPU for quick access and updates.
    current_length_data = torch.zeros(
        config.num_hidden_layers * 2, dtype=torch.long, device="cpu"
    )
    # Creating a KVCache for each pair of key and value in all layers
    past_key_values = [] * config.num_hidden_layers

    bias=0
    start_data_m=devices[0].index
    for i in range(config.num_hidden_layers):
        data_m=devices[i].index
        if data_m!=start_data_m:
            bias=0
            start_data_m=data_m
        try:
            past_key_values.append(
                [
                    KVCache(past_key_values_data_list[data_m-devices[0].index][2*bias + j], current_length_data[i * 2 + j])
                    for j in range(2)
                ]
            )
        except:
            past_key_values.append(
                [
                    KVCache(past_key_values_data_list[0][2 * bias + j],
                            current_length_data[i * 2 + j])
                    for j in range(2)
                ]
            )
        bias+=1
    return past_key_values, past_key_values_data_list, current_length_data


import torch

def initialize_eagle_past_key_values(model, max_length=2200, batch_size=1):
    """
    静态方法：为 Eagle 模型（Eagle2 或 Eagle3）初始化 KV Cache。
    
    融合了 Base Model 的设备感知能力和 Eagle 类的配置适配逻辑。
    
    Args:
        model: Eagle 模型实例 (Eagle2 或 Eagle3)
        max_length (int): Cache 最大长度
        batch_size (int): Batch size
        
    Returns:
        tuple: 
            - past_key_values (list[list[KVCache]]): 封装好的 Cache 对象列表
            - past_key_values_data_list (list[Tensor]): 实际存储数据的 Tensor 列表
            - current_length_data (Tensor): 长度追踪 Tensor (CPU)
    """
    config = model.config

    # ------------------------------------------------------------------
    # 1. 结构归一化 (Normalize Model Structure)
    # ------------------------------------------------------------------
    if hasattr(model, 'midlayer'):
        # Eagle 3: 单层结构
        layers = [model.midlayer]
    elif hasattr(model, 'layers'):
        # Eagle 2: 列表结构 (ModuleList)
        layers = model.layers
    else:
        raise AttributeError("Unknown Eagle architecture: missing 'midlayer' or 'layers'.")
    
    num_layers = len(layers)

    # ------------------------------------------------------------------
    # 2. 关键参数探测 (Robust Parameter Detection)
    # ------------------------------------------------------------------
    # [Dtype & Device] 从第一层权重的属性获取，比 model.dtype 更可靠
    ref_weight = layers[0].self_attn.q_proj.weight
    dtype = ref_weight.dtype
    
    # [Config] 兼容 Eagle2 (MHA) 和 Eagle3 (GQA)
    # 如果 config 中没有 num_key_value_heads，则默认为 num_attention_heads (MHA)
    num_heads = config.num_attention_heads
    num_kv_heads = getattr(config, "num_key_value_heads", num_heads)
    if num_kv_heads is None: num_kv_heads = num_heads
    
    head_dim = getattr(config, "head_dim", config.hidden_size // num_heads)

    # ------------------------------------------------------------------
    # 3. 显存预分配 (Memory Allocation with Device Awareness)
    # 逻辑：按设备分组分配内存块，减少碎片，支持跨设备部署
    # ------------------------------------------------------------------
    devices = [layer.self_attn.q_proj.weight.device for layer in layers]
    past_key_values_data_list = []
    
    # 遍历层设备，构建连续的内存块
    current_device = devices[0]
    layers_on_current_device = 0
    
    for i, device in enumerate(devices):
        # 如果设备变更（跨卡），或者是最后一层，则分配之前的内存块
        if device != current_device:
            _allocate_block(
                past_key_values_data_list, layers_on_current_device, 
                batch_size, num_kv_heads, max_length, head_dim, 
                current_device, dtype
            )
            current_device = device
            layers_on_current_device = 0
        
        layers_on_current_device += 1

    # 分配最后一个设备的剩余层
    _allocate_block(
        past_key_values_data_list, layers_on_current_device, 
        batch_size, num_kv_heads, max_length, head_dim, 
        current_device, dtype
    )

    # ------------------------------------------------------------------
    # 4. 长度追踪与对象封装 (Length Tracking & Object Wrapping)
    # ------------------------------------------------------------------
    # [Length Tensor] 必须在 CPU 上以便快速索引和更新
    current_length_data = torch.zeros(num_layers * 2, dtype=torch.long, device="cpu")
    
    past_key_values = []
    
    # 将扁平的 data_list 映射回每层的 KVCache 对象
    # data_list index 逻辑：只要设备变了，list index 就会 +1
    data_list_idx = 0 
    layer_idx_in_block = 0
    start_device_idx = devices[0].index if devices[0].index is not None else 0
    
    for i in range(num_layers):
        device_idx = devices[i].index if devices[i].index is not None else 0
        
        # 如果当前层设备与起始设备索引跨度超过了当前块，切换到下一个 data block
        # 注意：这里简化处理，假设 data_list 顺序与 devices 遍历顺序一致
        if i > 0 and devices[i] != devices[i-1]:
            data_list_idx += 1
            layer_idx_in_block = 0
            
        kv_block = past_key_values_data_list[data_list_idx]
        
        past_key_values.append([
            KVCache(kv_block[2 * layer_idx_in_block],     current_length_data[2 * i]),
            KVCache(kv_block[2 * layer_idx_in_block + 1], current_length_data[2 * i + 1])
        ])
        
        layer_idx_in_block += 1

    return past_key_values, past_key_values_data_list, current_length_data


def _allocate_block(data_list, num_layers, batch_size, num_kv_heads, max_length, head_dim, device, dtype):
    """辅助函数：分配显存块并追加到列表"""
    if num_layers > 0:
        data = torch.zeros(
            num_layers * 2, # Key 和 Value 各一个
            batch_size,
            num_kv_heads,
            max_length,
            head_dim,
            device=device,
            dtype=dtype,
        )
        data_list.append(data)



