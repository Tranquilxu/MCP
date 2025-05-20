import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.MLA import DeepseekAttention, DeepseekV2RMSNorm


class VisionPromptNetwork(nn.Module):
    def __init__(self):
        super(VisionPromptNetwork, self).__init__()
        self.c1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.c2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.c3 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.c4 = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(64)

    def forward(self, x):
        # print("x size", x.size())
        x = self.c1(x)
        x = self.bn1(x)
        x = F.leaky_relu(x, negative_slope=0.05)
        x = F.dropout2d(x, p=0.001)
        x = self.c2(x)
        x = self.bn2(x)
        x = F.leaky_relu(x, negative_slope=0.05)
        x = F.dropout2d(x, p=0.001)
        x = self.c3(x)
        x = self.bn3(x)
        x = F.leaky_relu(x, negative_slope=0.05)
        x = self.c4(x)
        return x


# class TextPromptNetwork(nn.Module):
#     def __init__(self, d_model=768, num_heads=8):
#         super(TextPromptNetwork, self).__init__()
#         self.d_model = d_model
#         self.num_heads = num_heads
#         self.attention_layer = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)
#         self.norm = nn.LayerNorm(d_model)
#         self.dropout = nn.Dropout()
#
#     def forward(self, src):
#         # src = self.position_encoding(src)
#         src2, _ = self.attention_layer(src, src, src)
#         # src = src + src2
#         # src = self.norm(src)
#         # src = self.dropout(src)
#
#         src = src + self.dropout(src2)
#         src = self.norm(src)
#
#         return src


class TextPromptNetwork(nn.Module):
    def __init__(self,
                 d_model=768,
                 num_heads=8,
                 q_lora_rank=64,  # Q的低秩维度
                 kv_lora_rank=64,  # K/V的低秩维度
                 qk_rope_head_dim=64,  # 旋转位置编码维度
                 max_position_embeddings=2048):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        # 根据DeepseekV2的配置参数计算相关维度
        self.qk_nope_head_dim = (d_model // num_heads) - qk_rope_head_dim
        self.v_head_dim = d_model // num_heads  # 保持与原始维度一致

        # 使用DeepseekV2的注意力机制
        self.attention_layer = DeepseekAttention(
            hidden_size=d_model,
            num_attention_heads=num_heads,
            q_lora_rank=q_lora_rank,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=self.v_head_dim,
            qk_nope_head_dim=self.qk_nope_head_dim,
            max_position_embeddings=max_position_embeddings,
            torch_dtype=torch.float32,
            attention_bias=False
        )

        self.norm = DeepseekV2RMSNorm(d_model)  # 使用DeepseekV2的RMSNorm
        self.dropout = nn.Dropout()

    def forward(self, src):
        # 生成位置ID (假设序列长度为src.shape[1])
        seq_len = src.size(1)
        position_ids = torch.arange(seq_len, dtype=torch.long, device=src.device)
        position_ids = position_ids.unsqueeze(0).expand(src.size(0), -1)

        # 压缩KV表示
        with torch.no_grad():  # KV压缩不参与梯度计算
            compressed_kv = self.attention_layer.compress_kv(
                hidden_states_kv=src,
                kv_position_ids=position_ids
            )

        # 注意力计算
        src2 = self.attention_layer(
            hidden_states_q=src,
            q_position_ids=position_ids,
            compressed_kv=compressed_kv
        )

        # 残差连接 + 归一化 + Dropout
        src = src + self.dropout(src2)
        src = self.norm(src)
        return src


class Adapter(nn.Module):
    def __init__(self, c_in, reduction=4):
        super(Adapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.fc(x)
        return x
