import torch


class KVCache:
    """
    A key-value cache for the model.

    This class provides a mechanism to maintain a growing cache of keys and values,
    particularly useful for models that benefit from caching previous states,
    like transformers during autoregressive decoding.

    Attributes:
        data (torch.Tensor): The tensor storing keys and values.
        current_length (int or torch.Tensor): Current length of the data being stored.
    """

    def __init__(self, data, current_length):
        """
        Initialize the KVCache.

        Args:
            data (torch.Tensor): Initial tensor to store the keys and values.
            current_length (int or torch.Tensor): Initial length of the data.
        """
        self.data = data
        if isinstance(current_length, int):
            self.current_length = torch.tensor(current_length, dtype=torch.long)
        else:
            self.current_length = current_length

    @property
    def shape(self):
        """Return the shape of the data tensor with updated length."""
        current_len = self.current_length.item() if isinstance(self.current_length, torch.Tensor) else self.current_length
        return (
            self.data.shape[0],
            self.data.shape[1],
            current_len,
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
        if isinstance(self.current_length, torch.Tensor):
            self.current_length.fill_(prev_length + tgt.shape[dim])
        else:
            self.current_length = prev_length + tgt.shape[dim]

    def cat(self, tensor: torch.Tensor, dim: int = 2):
        """
        Concatenate the given tensor with the current data.

        Args:
            tensor (torch.Tensor): The tensor to be concatenated.
            dim (int, optional): The dimension along which concatenation should be done. Default is 2.

        Returns:
            torch.Tensor: The data tensor after concatenation up to the current length.
        """
        current_len = self.current_length.item() if isinstance(self.current_length, torch.Tensor) else self.current_length
        dst = self.data.narrow(dim, current_len, tensor.shape[dim])
        dst.copy_(tensor, non_blocking=True)
        
        if isinstance(self.current_length, torch.Tensor):
            self.current_length.add_(tensor.shape[dim])
        else:
            self.current_length += tensor.shape[dim]
        
        new_len = self.current_length.item() if isinstance(self.current_length, torch.Tensor) else self.current_length
        return torch.narrow(self.data, dim, 0, new_len)
    
    def get_data(self):
        """
        Get the current valid data (up to current_length).
        
        Returns:
            torch.Tensor: Valid cached data.
        """
        current_len = self.current_length.item() if isinstance(self.current_length, torch.Tensor) else self.current_length
        return torch.narrow(self.data, 2, 0, current_len)
    
    def reset(self):
        """Reset the cache to empty state."""
        if isinstance(self.current_length, torch.Tensor):
            self.current_length.fill_(0)
        else:
            self.current_length = 0


def initialize_past_key_values(model, max_length=2200, batch_size=1):
    """
    Initialize past key and value states for a given transformer model.

    This function prepares key-value cache structures for the model, allowing it to store and reuse
    past key and value states during autoregressive decoding, which can improve efficiency.

    Args:
        model (nn.Module): The transformer model for which past key-value states need to be initialized.
        max_length (int): Maximum sequence length for cache.
        batch_size (int): Batch size.

    Returns:
        tuple:
            - past_key_values (list): A list of KVCache objects for each layer in the model.
            - past_key_values_data (list): List of tensors storing all keys and values.
            - current_length_data (torch.Tensor): A tensor tracking the current length of keys/values in the cache.
    """
    # Extracting configuration from the model
    config = model.config
    
    # Get device for each layer
    devices = []
    for i in range(config.num_hidden_layers):
        try:
            device = model.model.layers[i].self_attn.q_proj.weight.device
        except:
            device = model.layers[i].self_attn.q_proj.weight.device
        devices.append(device)
    
    past_key_values_data_list = []
    startnum = 0
    startdevice = devices[0]
    
    for id, i in enumerate(devices):
        if startdevice != i:
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
            startnum = 0
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
    # Keep on CPU for quick access and updates.
    current_length_data = torch.zeros(
        config.num_hidden_layers * 2, dtype=torch.long, device="cpu"
    )
    
    # Creating a KVCache for each pair of key and value in all layers
    past_key_values = []
    
    bias = 0
    start_data_m = devices[0].index if hasattr(devices[0], 'index') else 0
    
    for i in range(config.num_hidden_layers):
        data_m = devices[i].index if hasattr(devices[i], 'index') else 0
        if data_m != start_data_m:
            bias = 0
            start_data_m = data_m
        
        try:
            device_offset = data_m - (devices[0].index if hasattr(devices[0], 'index') else 0)
            past_key_values.append(
                [
                    KVCache(
                        past_key_values_data_list[device_offset][2 * bias + j],
                        current_length_data[i * 2 + j]
                    )
                    for j in range(2)
                ]
            )
        except:
            past_key_values.append(
                [
                    KVCache(
                        past_key_values_data_list[0][2 * bias + j],
                        current_length_data[i * 2 + j]
                    )
                    for j in range(2)
                ]
            )
        bias += 1
    
    return past_key_values, past_key_values_data_list, current_length_data

