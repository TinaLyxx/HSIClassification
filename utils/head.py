import torch
from torch import nn

class Head(nn.Module):
    def __init__(self, input_dim, num_classes, sequence_length, head='AP'):
        super().__init__()
        self.head = head
        self.pooling_head = ClassificationHead(input_dim, num_classes)
        self.fc_head = FC_ClassificationHead(input_dim, num_classes, sequence_length)
        self.attn_head = AttentionClassificationHead(input_dim, num_classes)

    def forward(self,x):
        if self.head == 'AP':
            x = self.pooling_head(x)
        elif self.head == 'FC':
            x = self.fc_head(x)
        elif self.head == 'AT':
            x = self.attn_head(x)
        else:
            raise NotImplementedError(
                f'classification head does not support {self.head}')
        return x

class ClassificationHead(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(ClassificationHead, self).__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)  # 使用全局平均池化，将 L 维度池化为 1
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        # 输入 x 形状为 (B, L, C)
        x = x.permute(0, 2, 1)  # 转换为 (B, C, L) 以适应池化
        x = self.pool(x)  # 池化到 (B, C, 1)
        x = x.squeeze(-1)  # 移除最后一个维度，形状变为 (B, C)
        x = self.fc(x)  # 全连接层输出，形状为 (B, num_classes)
        return x

class FC_ClassificationHead(nn.Module):
    def __init__(self, input_dim, num_classes, sequence_length):
        super(ClassificationHead, self).__init__()
        self.fc = nn.Linear(input_dim * sequence_length, num_classes)

    def forward(self, x):
        # 输入 x 形状为 (B, L, C)
        x = x.view(x.size(0), -1)  # 展平为 (B, L*C)
        x = self.fc(x)  # 输出形状为 (B, num_classes)
        return x

class AttentionClassificationHead(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(AttentionClassificationHead, self).__init__()
        self.attention = nn.Linear(input_dim, 1)  # 简单的注意力层
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        # 输入 x 形状为 (B, L, C)
        attn_weights = torch.softmax(self.attention(x), dim=1)  # (B, L, 1)
        x = (x * attn_weights).sum(dim=1)  # 加权求和，得到 (B, C)
        x = self.fc(x)  # 输出形状为 (B, num_classes)
        return x