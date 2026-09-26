import torch
import tiktoken
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        super().__init__()

        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        ## Implementing sliding window
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length]
            self.input_ids.append(input_chunk)
            self.target_ids.append(target_chunk)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        return torch.tensor(self.input_ids[index]), torch.tensor(self.target_ids[index])



def create_dataloader_v1(txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0):
    tokenizer = tiktoken.get_encoding('gpt2')

    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloder = DataLoader(
        dataset,
        batch_size = batch_size,
        shuffle=shuffle,
        drop_last = drop_last,
        num_workers=num_workers
    )
    return dataloder


class MultiHeadAttention(nn.Module):
    def __init__(self, in_d, out_d, context_length, dropout, n_heads, qkv_bias = False):
        super().__init__()
        assert(out_d % n_heads == 0)

        self.d_out = out_d
        self.n_heads = n_heads
        self.head_dim = out_d // n_heads

        self.w_query = nn.Linear(in_d, out_d, bias=qkv_bias)
        self.w_key = nn.Linear(in_d, out_d, bias=qkv_bias)
        self.w_value = nn.Linear(in_d, out_d, bias=qkv_bias)
        self.out_proj = nn.Linear(out_d, out_d)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, inputs):
        b, num_tokens, d_in = inputs.shape

        query = self.w_query(inputs)
        keys = self.w_key(inputs)
        values = self.w_value(inputs)

        keys = keys.view(b, num_tokens, self.n_heads, self.head_dim)
        values = values.view(b, num_tokens, self.n_heads, self.head_dim)
        queries = query.view(b, num_tokens, self.n_heads, self.head_dim)

        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        queries = queries.transpose(1, 2)

        attn_score = queries @ keys.transpose(2, 3)
        attn_score.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        attn_weight = torch.softmax(attn_score / keys.shape[-1] ** 0.5, dim=-1)
        attn_weight = self.dropout(attn_weight)

        context_vec = (attn_weight @ values).transpose(1, 2)

        context_vec = context_vec.reshape(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)

        return context_vec