import torch
import torch.nn as nn

class LinearProjector(nn.Module):
    """
    线性投影器: 视觉特征 -> 语言模型嵌入空间
    架构: Linear(input_dim, hidden_dim) -> GELU -> Linear(hidden_dim, output_dim)
    """

    def __init__(self, input_dim: int = 768, output_dim: int = 2048):
        """
        初始化投影器

        Args:
            input_dim: 输入维度 (视觉特征维度)
            output_dim: 输出维度 (LLM嵌入维度)
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        # 线性变换
        self.linear = nn.Linear(input_dim, output_dim)

        # 初始化权重
        self._init_weights()

        print(f"Linear Projector: {input_dim} -> {output_dim}")
        print(f"Trainable params: {self.get_trainable_params():,}")

    def _init_weights(self):
        """初始化权重"""
        # 使用Xavier初始化
        nn.init.xavier_uniform_(self.linear.weight)

        # 偏置初始化为0
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            vision_features: 视觉特征 (B, seq_len, input_dim)
                             通常seq_len=197 (196 patches + 1 cls)

        Returns:
            投影后的特征 (B, seq_len, output_dim)
        """
        # 第一层: 线性变换 + 激活
        output = self.linear(vision_features)
        return output

    def get_trainable_params(self) -> int:
        """获取可训练参数数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

class MLPProjector(nn.Module):
    """
    MLP投影器
    输入:
        vision_features: [B, N, 768]
    输出:
        projected_features: [B, N, 2048]
    """
    def __init__(self, input_dim: int = 768, hidden_dim: list = 1536,
                 output_dim: int = 2048, activation: str = "gelu", dropout: float = 0.0):
        """
        初始化MLP投影器

        Args:
            input_dim: 输入维度
            hidden_dim: 隐藏层维度
            output_dim: 输出维度
            activation: 激活函数
            dropout: dropout概率
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.activation_name = activation
        self.dropout = dropout

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = self._get_activation(activation)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

        # 初始化权重
        self._init_weights()

        print(f"MLP Projector: {input_dim} -> {hidden_dim} -> {output_dim}")
    def _get_activation(self, activation: str) -> nn.Module:
        """获取激活函数"""
        if activation == "gelu":
            return nn.GELU()
        elif activation == "relu":
            return nn.ReLU()
        elif activation == "silu":
            return nn.SiLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def _init_weights(self):
        for module in [self.fc1, self.fc2]:
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        if vision_features.size(-1) != self.input_dim:
            raise ValueError(
                f"Expected input last dim = {self.input_dim}, "
                f"got {vision_features.size(-1)}"
            )

        x = self.fc1(vision_features)   # [B, N, 1536]
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)                 # [B, N, 2048]
        return x

    def get_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

class DeepMLPProjector(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=1536, output_dim=2048, dropout=0.05):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, output_dim)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        if x.size(-1) != self.input_dim:
            raise ValueError(
                f"Expected last dim {self.input_dim}, got {x.size(-1)}"
            )
        return self.net(x)
    def get_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
if __name__ == "__main__":
    # projector = LinearProjector(
    #     input_dim=768,
    #     output_dim=2048,
    # activation="gelu",
    # )
    # projector = MLPProjector(
    #     input_dim=768,
    #     hidden_dim=1536,
    #     output_dim=2048,
    #     activation="gelu",
    #     dropout=0.0,
    # )
    projector = DeepMLPProjector(
        input_dim=768,
        hidden_dim=1536,
        output_dim=2048,
        dropout=0.0,
    )
    vision_features = torch.randn(2, 196, 768)   # [B, N, C]
    out = projector(vision_features)

    print(projector)
    print("input shape :", vision_features.shape)
    print("output shape:", out.shape)
    print("trainable params:", projector.get_trainable_params())