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


class CausalSelfAttention(nn.Module):
    def __init__(self, in_d, out_d, dropout, context_length, qkv_bias = False):
        super().__init__()
        self.w_query = torch.nn.Linear(in_d, out_d, bias=qkv_bias)
        self.w_key = torch.nn.Linear(in_d, out_d, bias=qkv_bias)
        self.w_value = torch.nn.Linear(in_d, out_d, bias=qkv_bias)
        self.dropout = torch.nn.Dropout(dropout)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))


    def forward(self, inputs):
        b, num_token, d_in = inputs.shape

        query = self.w_query(inputs)
        key = self.w_key(inputs)
        value = self.w_value(inputs)

        attn_scores = query @ key.transpose(1, 2)
        attn_scores.masked_fill_(self.mask.bool()[:num_token, :num_token], -torch.inf)
        attn_weight = torch.softmax(attn_scores / key.shape[-1]**0.5, dim=1)
        attn_weight = self.dropout(attn_weight)

        context_vec = attn_weight @ value
        return context_vec


class MultiHeadAttention(nn.Module):

    def __init__(self, in_d, out_d, dropout, context_len, head, qkv_bias=False):
        super().__init__()
        self.heads = torch.nn.ModuleList(
            [CausalSelfAttention(in_d, out_d, dropout, context_len, qkv_bias) for _ in range(head)]
        )

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)