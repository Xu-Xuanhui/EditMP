import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from models_search.ViT_helper import DropPath, to_2tuple, trunc_normal_
#from models_search.diff_aug import DiffAugment
import torch.utils.checkpoint as checkpoint


class matmul(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1, x2):
        x = x1 @ x2
        return x


def count_matmul(m, x, y):
    num_mul = x[0].numel() * x[1].size(-1)
    # m.total_ops += torch.DoubleTensor([int(num_mul)])
    m.total_ops += torch.DoubleTensor([int(0)])


class PixelNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()

    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input ** 2, dim=2, keepdim=True) + 1e-8)
class ChanNorm(nn.Module):  #论文中AdaIN的过程
    # AdaIN = y_(s,i)*((x - mu(x))/sigma(x)) + y_(b,i) AdaIN公式
    def __init__(self, steps, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, steps, 1))
        self.b = nn.Parameter(torch.zeros(1, steps, 1))

    def forward(self, x):
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b

def gelu(x):
    """ Original Implementation of the gelu activation function in Google Bert repo when initialy created.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def leakyrelu(x):
    return nn.functional.leaky_relu_(x, 0.2)


class CustomAct(nn.Module):
    def __init__(self, act_layer):
        super().__init__()
        if act_layer == "gelu":
            self.act_layer = gelu
        elif act_layer == "leakyrelu":
            self.act_layer = leakyrelu

    def forward(self, x):
        return self.act_layer(x)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer='gelu', drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = CustomAct(act_layer)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=3, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = 1

        head_dim = dim // num_heads

        # head_dim = 1

        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.mat = matmul()

    def forward(self, x):
        B, N, C = x.shape
        # print(x.size())
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # print('qkv',qkv.size())
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        attn = (self.mat(q, k.transpose(-2, -1))) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = self.mat(attn, v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CustomNorm(nn.Module):
    def __init__(self, norm_layer, dim):
        super().__init__()

        self.norm_type = norm_layer
        if norm_layer == "ln":
            self.norm = nn.LayerNorm(dim)
        elif norm_layer == "bn":
            self.norm = nn.BatchNorm1d(dim)
            # print('BatchNorm1d')
        elif norm_layer == "in":
            self.norm = nn.InstanceNorm1d(dim)
        elif norm_layer == "pn":
            self.norm = PixelNorm(dim)

    def forward(self, x):
        # if self.norm_type == "bn" or self.norm_type == "in":
        #     x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1)
        #     return x
        # elif self.norm_type == "none":
        #     return x
        # else:

        return self.norm(x)


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=gelu, norm_layer=nn.LayerNorm, if_G=True):
        super().__init__()
        if if_G:
            self.norm1 = CustomNorm(norm_layer, 8)
        else:
            self.norm1 = CustomNorm(norm_layer, 9)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # self.norm2 = CustomNorm(norm_layer, dim)
        if if_G:
            self.norm2 = CustomNorm(norm_layer, 8)
        else:
            self.norm2 = CustomNorm(norm_layer, 9)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class StageBlock(nn.Module):
    def __init__(self, depth, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=gelu, norm_layer=nn.LayerNorm):
        super().__init__()
        self.depth = depth
        models = [Block(
            dim=dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop,
            attn_drop=attn_drop,
            drop_path=drop_path,
            act_layer=act_layer,
            norm_layer=norm_layer,
            if_G=True
        ) for i in range(depth)]
        self.block = nn.Sequential(*models)

    def forward(self, x):
        #         for blk in self.block:
        #             # x = blk(x)
        #             checkpoint.checkpoint(blk, x)
        #         x = checkpoint.checkpoint(self.block, x)
        x = self.block(x)

        return x


def pixel_upsample(x, H, W):
    B, N, C = x.size()
    assert N == H * W
    x = x.permute(0, 2, 1)
    x = x.view(-1, C, H, W)
    x = nn.PixelShuffle(2)(x)
    B, C, H, W = x.size()
    x = x.view(-1, C, H * W)
    x = x.permute(0, 2, 1)
    return x, H, W


def bicubic_upsample(x, H, W):
    B, N, C = x.size()
    assert N == H * W
    x = x.permute(0, 2, 1)
    x = x.view(-1, C, H, W)
    x = nn.functional.interpolate(x, scale_factor=2, mode='bicubic')
    B, C, H, W = x.size()
    x = x.view(-1, C, H * W)
    x = x.permute(0, 2, 1)
    return x, H, W

class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim, lr_mul = 1, bias = True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim))

        self.lr_mul = lr_mul

    def forward(self, input):
        return F.linear(input, self.weight * self.lr_mul, bias=self.bias * self.lr_mul)
def leaky_relu(p=0.2):
    return nn.LeakyReLU(p, inplace=True)
class StyleVectorizer(nn.Module):
    def __init__(self, args,emb = 8, depth = 2, lr_mul = 0.1):
        super(StyleVectorizer,self).__init__()

        layers = []
        for i in range(depth):
            layers.extend([EqualLinear(emb, emb, lr_mul), leaky_relu()])

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = F.normalize(x, dim=1)
        return self.net(x)

class Generator(nn.Module):

    def __init__(self, args, embed_dim=3, num_heads=1, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0.,drop_path_rate=0., hybrid_backbone=None, norm_layer='bn'):
        super(Generator, self).__init__()
        """
            Args:
                steps (int): Length of action sequences (Default = 10)
                embed_dim (int): Number of robot arm joint angles (Default = 10) or 3D Coordinates (Default = 3)
                num_heads (int): Number of Attention heads 

        """
        self.args = args
        self.ch = embed_dim
        self.steps = args.steps
        self.embed_dim = embed_dim = args.gf_dim

        norm_layer = args.g_norm
        mlp_ratio = args.g_mlp
        depth = [int(i) for i in args.g_depth.split(",")]
        # print(depth[0])
        self.initial_block = nn.Parameter(torch.randn((1, self.steps, self.embed_dim)))
        act_layer = args.g_act

        self.act_latent = CustomAct(act_layer='leakyrelu')
        self.w2s_1 = nn.Linear(args.latent_dim, self.steps * self.embed_dim)


        # self.pos_embed = nn.Parameter(torch.zeros(1, self.steps, self.embed_dim))
        self.pos_embed = torch.eye(self.steps).repeat(args.gen_batch_size, 1, 1).cuda()
        self.tanh = nn.Tanh()
        self.blocks_1 = StageBlock(
            depth=depth[0],
            dim=embed_dim + self.steps,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=0,
            act_layer=act_layer,
            norm_layer=norm_layer
        )

        self.blocks_2 = StageBlock(
            depth=depth[1],
            dim=embed_dim + self.steps,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=0,
            act_layer=act_layer,
            norm_layer=norm_layer
        )

        self.blocks_3 = StageBlock(
            depth=depth[2],
            dim=embed_dim + self.steps,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=0,
            act_layer=act_layer,
            norm_layer=norm_layer
        )

        # self.deconv = nn.Sequential(
        #     nn.Conv2d(in_channels=1, out_channels=1, kernel_size=5, stride=3),
        #
        # )
        self.output =nn.Linear((embed_dim + self.steps)*self.steps, embed_dim * self.steps)
    #         self.apply(self._init_weights)

    #     def _init_weights(self, m):
    #         if isinstance(m, nn.Linear):
    #             trunc_normal_(m.weight, std=.02)
    #             if isinstance(m, nn.Linear) and m.bias is not None:
    #                 nn.init.constant_(m.bias, 0)
    #         elif isinstance(m, nn.Conv2d):
    #             trunc_normal_(m.weight, std=.02)
    #             if isinstance(m, nn.Conv2d) and m.bias is not None:
    #                 nn.init.constant_(m.bias, 0)
    #         elif isinstance(m, nn.LayerNorm):
    #             nn.init.constant_(m.bias, 0)
    #             nn.init.constant_(m.weight, 1.0)
    #         elif isinstance(m, nn.BatchNorm1d):
    #             nn.init.constant_(m.bias, 0)
    #             nn.init.constant_(m.weight, 1.0)
    #         elif isinstance(m, nn.InstanceNorm1d):
    #             nn.init.constant_(m.bias, 0)
    #             nn.init.constant_(m.weight, 1.0)
    def forward(self, w, epoch):
        # if self.args.latent_norm:
        #     latent_size = z.size(-1)
        #     z = (z / z.norm(dim=-1, keepdim=True) * (latent_size ** 0.5))
        # print(z)
        

        x = self.initial_block.expand(w.shape[0], -1, -1)
        s_1 = self.w2s_1(w).view(-1, self.steps, self.embed_dim)
        x = self.act_latent(x + s_1)
        x = torch.cat((x, self.pos_embed), 2)

        output = self.blocks_1(x)
        # output = self.AdaIN_1(output)

        # s_1 = torch.cat((s_1, self.pos_embed), 2)


        output = self.blocks_2(output)
        output = self.blocks_3(output)

        output = self.output(output.view(-1,self.steps* (self.steps + self.embed_dim)))

        output = self.tanh(output) * math.pi

        return output.view(-1, self.steps, self.embed_dim)

        # return output[:, :, :6]
class Orthotics(nn.Module):

    def __init__(self, args, embed_dim=3, num_heads=1, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0.,drop_path_rate=0., hybrid_backbone=None, norm_layer='bn'):
        super(Orthotics, self).__init__()
        """
            Args:
                steps (int): Length of action sequences (Default = 10)
                embed_dim (int): Number of robot arm joint angles (Default = 10) or 3D Coordinates (Default = 3)
                num_heads (int): Number of Attention heads 

        """
        self.args = args
        self.ch = embed_dim
        self.steps = args.steps
        self.embed_dim = embed_dim = args.gf_dim

        norm_layer = args.g_norm
        mlp_ratio = args.g_mlp
        depth = [int(i) for i in args.g_depth.split(",")]
        # print(depth[0])
        act_layer = args.g_act

        self.l1 = nn.Linear(args.latent_dim, self.steps * self.embed_dim)
        self.initial_block = nn.Parameter(torch.randn((1, self.steps, self.embed_dim)))
        # self.pos_embed = nn.Parameter(torch.zeros(1, self.steps, self.embed_dim))
        self.pos_embed = torch.eye(self.steps).repeat(args.gen_batch_size, 1, 1).cuda()
        self.tanh = nn.Tanh()
        self.blocks_1 = StageBlock(
            depth=depth[0],
            dim=embed_dim + self.steps,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=0,
            act_layer=act_layer,
            norm_layer=norm_layer
        )

        self.output =nn.Linear((embed_dim + self.steps)*self.steps, embed_dim * self.steps)

    def forward(self, x, epoch):

        x = torch.cat((x, self.pos_embed), 2)
        # B = x.size()
        # H, W = self.bottom_width, self.bottom_width
        output = self.blocks_1(x)
        output = self.output(output.view(-1,self.steps* (self.steps + self.embed_dim)))
        output = self.tanh(output) * 3.1415
        return output.view(-1, self.steps, self.embed_dim)

class DisBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=leakyrelu, norm_layer=nn.LayerNorm, separate=0):
        super().__init__()
        self.norm1 = CustomNorm(norm_layer, 8)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = CustomNorm(norm_layer, 8)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.gain = np.sqrt(0.5) if norm_layer == "none" else 1

    def forward(self, x):
        x = x * self.gain + self.drop_path(self.attn(self.norm1(x))) * self.gain
        x = x * self.gain + self.drop_path(self.mlp(self.norm2(x))) * self.gain
        return x


class Discriminator(nn.Module):
    def __init__(self, args, num_classes=1, embed_dim=None, depth=7, num_heads=2, mlp_ratio=4., qkv_bias=False,
                 qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer=nn.LayerNorm):
        super().__init__()
        """
            Args:
                steps (int): Length of action sequences (Default = 8)
                embed_dim (int): Number of robot arm joint angles (Default = 6) or 3D Coordinates (Default = 3)
                num_heads (int): Number of Attention heads 

        """
        self.num_classes = num_classes
        self.num_features = embed_dim = self.embed_dim = args.df_dim
        self.steps = args.steps
        depth = args.d_depth
        self.args = args
        norm_layer = args.d_norm
        act_layer = args.d_act

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim + self.steps))
        # self.pos_embed = nn.Parameter(torch.zeros(1, self.steps, self.embed_dim))
        self.pos_embed = torch.eye(self.steps).repeat(args.gen_batch_size, 1, 1).cuda()

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks_1 = nn.ModuleList([
            DisBlock(
                dim=embed_dim + self.steps, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=0, act_layer=act_layer, norm_layer=norm_layer)
            for i in range(depth + 1)])

        self.last_block = nn.Sequential(
            Block(
                dim=embed_dim + self.steps, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], act_layer=act_layer, norm_layer=norm_layer,if_G=False)
        )

        self.norm = CustomNorm(norm_layer, self.steps+1)
        self.head = nn.Linear(embed_dim + self.steps, num_classes) if num_classes > 0 else nn.Identity()

        # trunc_normal_(self.pos_embed, std=.02)

        # trunc_normal_(self.cls_token, std=.02)
        # self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        # if "None" not in self.args.diff_aug:
        #     x = DiffAugment(x, self.args.diff_aug, True)
        # print(x.unsqueeze(-1).shape, self.pos_embed.shape)

        x = torch.cat((x, self.pos_embed), 2)
        # x = self.pos_drop(x)
        # print(x.size())
        B, _, C = x.shape
        # print(x.size())
        for blk in self.blocks_1:
            x = blk(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        # print('x shape', x.size(), 'cls_tokens shape', cls_tokens.size())
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.last_block(x)
        x = self.norm(x)

        return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
        # print(x.size())
        x = self.head(x)
        return x

