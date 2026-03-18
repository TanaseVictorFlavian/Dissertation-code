import torch
from slstm import sLSTMBlock
from mlstm import mLSTMBlock

def test_slstm():
    print("Testing sLSTMBlock...")
    batch_size = 2
    seq_len = 8
    dim = 16
    num_heads = 2
    
    model = sLSTMBlock(dim, num_heads)
    x = torch.randn(batch_size, seq_len, dim)
    output = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    assert output.shape == x.shape
    print("sLSTMBlock test passed!\n")

def test_mlstm():
    print("Testing mLSTMBlock...")
    batch_size = 2
    seq_len = 8
    dim = 16
    num_heads = 2
    
    model = mLSTMBlock(dim, num_heads)
    x = torch.randn(batch_size, seq_len, dim)
    output = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    assert output.shape == x.shape
    print("mLSTMBlock test passed!\n")

if __name__ == "__main__":
    test_slstm()
    test_mlstm()
