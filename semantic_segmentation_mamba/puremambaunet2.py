
from rs_mamba_ss import *
class RSM_SSv201(nn.Module):
    def __init__(
            self,
            patch_size=4,  # 补丁大小，表示图像分块的大小
            in_chans=4,  # 输入通道数，这里为4，表示输入图像有4个通道
            num_classes=1,  # 类别数，通常用于分类任务，原来这写的是1000
            depths=[2, 2, 9, 2],  # 每个阶段中的层数
            dims=[96, 192, 384, 768],  # 每个阶段的通道维度
            # =========================
            ssm_d_state=16,  # SSM状态维度
            ssm_ratio=2.0,  # SSM的比率参数
            ssm_dt_rank="auto",  # SSM的时间维度秩，自动调整
            ssm_act_layer="silu",  # SSM的激活函数类型，可以是"silu", "gelu", "relu"
            ssm_conv=3,  # SSM中卷积核的大小
            ssm_conv_bias=True,  # SSM中卷积层是否使用偏置项
            ssm_drop_rate=0.0,  # SSM的dropout率
            ssm_init="v0",  # SSM的初始化方式
            forward_type="v2",  # SSM的前向传播类型
            # =========================
            mlp_ratio=4.0,  # MLP的扩展比率
            mlp_act_layer="gelu",  # MLP的激活函数类型
            mlp_drop_rate=0.0,  # MLP的dropout率
            # =========================
            drop_path_rate=0,  # 随机深度丢弃率
            patch_norm=True,  # 是否对补丁进行归一化处理
            norm_layer="LN",  # 归一化层的类型，可以选择"LN"或"BN"
            use_checkpoint=False,  # 是否使用检查点机制
            **kwargs,  # 其他扩展参数
    ):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        if isinstance(dims, int):
            dims = [int(dims * 2 ** i_layer) for i_layer in range(self.num_layers)]
        self.num_features = dims[-1]
        self.dims = dims
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            bn=nn.BatchNorm2d,
        )

        _ACTLAYERS = dict(
            silu=nn.SiLU,
            gelu=nn.GELU,
            relu=nn.ReLU,
            sigmoid=nn.Sigmoid,
        )

        if isinstance(norm_layer, str) and norm_layer.lower() in ["ln"]:
            norm_layer: nn.Module = _NORMLAYERS[norm_layer.lower()]

        if isinstance(ssm_act_layer, str) and ssm_act_layer.lower() in ["silu", "gelu", "relu"]:
            ssm_act_layer: nn.Module = _ACTLAYERS[ssm_act_layer.lower()]

        if isinstance(mlp_act_layer, str) and mlp_act_layer.lower() in ["silu", "gelu", "relu"]:
            mlp_act_layer: nn.Module = _ACTLAYERS[mlp_act_layer.lower()]

        _make_patch_embed = self._make_patch_embed_v2
        self.patch_embed = _make_patch_embed(in_chans, dims[0], patch_size, patch_norm, norm_layer)

        _make_downsample = self._make_downsample_v3

        # self.encoder_layers = [nn.ModuleList()] * self.num_layers
        self.encoder_layers = []
        self.decoder_layers = []

        for i_layer in range(self.num_layers):
            # downsample = _make_downsample(
            #     self.dims[i_layer],
            #     self.dims[i_layer + 1],
            #     norm_layer=norm_layer,
            # ) if (i_layer < self.num_layers - 1) else nn.Identity()

            downsample = _make_downsample(
                self.dims[i_layer - 1],
                self.dims[i_layer],
                norm_layer=norm_layer,
            ) if (i_layer != 0) else nn.Identity()  # ZSJ 修改为i_layer != 0，也就是第一层不下采样，和论文的图保持一致，也方便我取出每个尺度处理好的特征

            self.encoder_layers.append(self._make_layer(
                dim=self.dims[i_layer],
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                use_checkpoint=use_checkpoint,
                norm_layer=norm_layer,
                downsample=downsample,
                # =================
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_dt_rank=ssm_dt_rank,
                ssm_act_layer=ssm_act_layer,
                ssm_conv=ssm_conv,
                ssm_conv_bias=ssm_conv_bias,
                ssm_drop_rate=ssm_drop_rate,
                ssm_init=ssm_init,
                forward_type=forward_type,
                # =================
                mlp_ratio=mlp_ratio,
                mlp_act_layer=mlp_act_layer,
                mlp_drop_rate=mlp_drop_rate,
            ))
            if i_layer != 0:
                self.decoder_layers.append(
                    Decoder_Block(in_channel=self.dims[i_layer], out_channel=self.dims[i_layer - 1]))

        self.encoder_block1, self.encoder_block2, self.encoder_block3, self.encoder_block4 = self.encoder_layers
        self.deocder_block1, self.deocder_block2, self.deocder_block3 = self.decoder_layers

        self.upsample_x4 = nn.Sequential(
            nn.Conv2d(self.dims[0], self.dims[0] // 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.dims[0] // 2),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(self.dims[0] // 2, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(scale_factor=2)
        )
        self.conv_out_seg = nn.Conv2d(8, 1, kernel_size=7, stride=1, padding=3)

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @staticmethod
    def _make_patch_embed_v2(in_chans=4, embed_dim=96, patch_size=4, patch_norm=True,
                             norm_layer=nn.LayerNorm):  # 修改图片通道数在这里改
        assert patch_size == 4
        return nn.Sequential(
            nn.Conv2d(in_chans, embed_dim // 2, kernel_size=3, stride=1, padding=1),
            (Permute(0, 2, 3, 1) if patch_norm else nn.Identity()),
            (norm_layer(embed_dim // 2) if patch_norm else nn.Identity()),
            (Permute(0, 3, 1, 2) if patch_norm else nn.Identity()),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, stride=2, padding=1),
            Permute(0, 2, 3, 1),
            (norm_layer(embed_dim) if patch_norm else nn.Identity()),
        )
    # 其中图像首先被划分为多个 patch，然后每个 patch 被线性投影到一个高维空间中，以供 Transformer 模型处理。
    @staticmethod
    def _make_downsample_v3(dim=96, out_dim=192, norm_layer=nn.LayerNorm):
        return nn.Sequential(
            Permute(0, 3, 1, 2),
            nn.Conv2d(dim, out_dim, kernel_size=3, stride=2, padding=1),
            Permute(0, 2, 3, 1),
            norm_layer(out_dim),
        )

    @staticmethod
    def _make_layer(
            dim=96,
            drop_path=[0, 0],
            use_checkpoint=False,
            norm_layer=nn.LayerNorm,
            downsample=nn.Identity(),
            # ===========================
            ssm_d_state=16,
            ssm_ratio=2.0,
            ssm_dt_rank="auto",
            ssm_act_layer=nn.SiLU,
            ssm_conv=3,
            ssm_conv_bias=True,
            ssm_drop_rate=0.0,
            ssm_init="v0",
            forward_type="v2",
            # ===========================
            mlp_ratio=4.0,
            mlp_act_layer=nn.GELU,
            mlp_drop_rate=0.0,
            **kwargs,
    ):
        depth = len(drop_path)
        blocks = []
        for d in range(depth):
            blocks.append(OSSBlock(
                hidden_dim=dim,
                drop_path=drop_path[d],
                norm_layer=norm_layer,
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_dt_rank=ssm_dt_rank,
                ssm_act_layer=ssm_act_layer,
                ssm_conv=ssm_conv,
                ssm_conv_bias=ssm_conv_bias,
                ssm_drop_rate=ssm_drop_rate,
                ssm_init=ssm_init,
                forward_type=forward_type,
                mlp_ratio=mlp_ratio,
                mlp_act_layer=mlp_act_layer,
                mlp_drop_rate=mlp_drop_rate,
                use_checkpoint=use_checkpoint,
            ))

        return nn.Sequential(OrderedDict(
            # ZSJ 把downsample放到前面来，方便我取出encoder中每个尺度处理好的图像，而不是刚刚下采样完的图像
            downsample=downsample,
            blocks=nn.Sequential(*blocks, ),
        ))

    def forward(self, x1: torch.Tensor):  # 输入, 256x256, 4个通道

        x1 = self.patch_embed(x1)  # 64x64, 96个通道

        x1_1 = self.encoder_block1(x1)  # 64x64, 96个通道
        x1_2 = self.encoder_block2(x1_1)  # 32x32, 192个通道
        x1_3 = self.encoder_block3(x1_2)  # 16x16, 384个通道
        x1_4 = self.encoder_block4(x1_3)  # 8x8, 768个通道

        # 在通过编码器后，特征图的排列可能不符合解码器的输入要求，因此需要进行重排
        x1_1 = rearrange(x1_1, "b h w c -> b c h w").contiguous()
        x1_2 = rearrange(x1_2, "b h w c -> b c h w").contiguous()
        x1_3 = rearrange(x1_3, "b h w c -> b c h w").contiguous()
        x1_4 = rearrange(x1_4, "b h w c -> b c h w").contiguous()

        decode_3 = self.deocder_block3(x1_4, x1_3)  # 16x16, 384个通道
        decode_2 = self.deocder_block2(decode_3, x1_2)  # 32x32, 192个通道
        decode_1 = self.deocder_block1(decode_2, x1_1)  # 64x64, 96个通道

        output = self.upsample_x4(decode_1)  # 256x256, 8个通道
        output = self.conv_out_seg(output)  # 输出 256x256, 1个通道

        return output

