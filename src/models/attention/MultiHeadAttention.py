import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout=0.1) -> None:
        super(MultiHeadAttention, self).__init__()
        '''
        :param d_model: Dimension of the attention matrix
        :param num_heads: Number of attention heads in the multi-head attention
        :param dropout: Dropout value
        :return: None
        '''
        self.d_model = d_model
        self.dropout_val = dropout
        self.dropout = nn.Dropout(dropout)
        self.num_heads = num_heads
        self.head_shape = int(d_model / num_heads)

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        # Defining the weight matrices for the query, key, value and attention layers
        self.w_q = nn.Linear(d_model, d_model, bias = False)
        self.w_k = nn.Linear(d_model, d_model, bias = False)
        self.w_v = nn.Linear(d_model, d_model, bias = False)
        self.w_o = nn.Linear(d_model, d_model, bias = False)

    @staticmethod
    def calculate_attention(q: torch.Tensor, v: torch.Tensor, k: torch.Tensor,
                            m: torch.Tensor, dropout: float = None) -> torch.Tensor:
        '''
        Calculates the attention value using the query, key and value. Returns the attention and the softmax
        activation values.

        Paper: "Attention is all you need" -> https://arxiv.org/pdf/1706.03762.pdf
        Formula 1, page 4

        :param q: Queries
        :param v: Values
        :param k: Keys
        :param m: Masking Matrix
        :param dropout: Dropout value
        :return: attention, softmax activation values
        Output shape: (batch_size, num_heads, sequence length, head shape)
        '''
        # calculate Q dot V
        batch, num_heads, seq_len, head_shape = q.shape
        logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_shape)

        # mask values to make the model causal
        if m is not None:
            logits = logits.masked_fill_(m == 0, -1e9)

        # calculate softmax and dropout
        sftmx_values = F.softmax(logits, dim=-1)
        if dropout is not None:
            sftmx_values = F.dropout(sftmx_values, p=dropout, training=True)

        # finally calculate the attention
        attention = torch.matmul(sftmx_values, v)

        return attention, sftmx_values

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        '''
        :param q: queries (batch_size, sequence length, d_model)
        :param k: keys (batch_size, sequence length, d_model)
        :param v: values batch_size, sequence length, d_model)
        :param mask: mask (seq_len, seq_len)
        :return: attention values (batch_size, sequence length, d_model ), softmax activation values
        (batch_size, sequence length, d_model)
        '''

        # multiply queries, key and values with their respective weight matrices
        q_mk = self.w_q(q) # out size : (batch_sz, sequence length, d_model)
        k_mk = self.w_k(k) # out size : (batch_sz, sequence length, d_model)
        v_mk = self.w_v(v) # out size : (batch_sz, sequence length, d_model)


        # divide the model into num_heads heads, each head has a size of d_model/num_heads
        # Transpose the first and second dimension to get size: (batch size, num_heads, sequence length, head shape)
        batch, seq_len, d_model = q_mk.shape
        q_mk = q_mk.view(batch, seq_len, self.num_heads, self.head_shape).permute(0, 2, 1, 3)
        k_mk = k_mk.view(k_mk.shape[0], k_mk.shape[1], self.num_heads, self.head_shape).permute(0, 2, 1, 3)
        v_mk = v_mk.view(v_mk.shape[0], v_mk.shape[1], self.num_heads, self.head_shape).permute(0, 2, 1, 3)

        # calculate the attention value
        attention, sftmx_values = MultiHeadAttention.calculate_attention(q_mk, v_mk, k_mk,
                                                                         m=mask, dropout = self.dropout_val)

        # merge the value of each attention head into get the final output, multiply with the
        # w_o weight matrix to get the final output
        # from shape:  (batch size, num_heads, sequence length, head_shape) to shape
        # (batch size, sequence length, d_model)
        attention = attention.permute(0, 2, 1, 3).contiguous().view(batch, seq_len, d_model)
        attention = self.w_o(attention)


        return attention




