import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Function
import math
from einops import rearrange, repeat

from mamba_ssm import Mamba2
import tree_generate
from timm.models.layers import trunc_normal_, DropPath

from utils.util_mst import MinimumSpanningTree
from utils.pos_embed import get_2d_sincos_pos_embed
from utils.attention_layer import CrossAttention, Spa_Attention, Spe_Attention
from utils.head import Head


class _BFS(Function):
    @staticmethod
    def forward(ctx, edge_index, max_adj_per_vertex, root):
        sorted_index, sorted_parent, sorted_child =\
                tree_generate.bfs_forward(edge_index, max_adj_per_vertex, root)
        return sorted_index, sorted_parent, sorted_child


class PatchEmbed(nn.Module):
    def __init__(self, in_channels, hidden_dim=64, group_num=4, group_norm=True):
        super().__init__()
        self.group_norm = group_norm
        self.patch_embedding = nn.Sequential(nn.Conv2d(in_channels=in_channels,out_channels=hidden_dim,kernel_size=1,stride=1,padding=0),
                                             nn.GroupNorm(group_num,hidden_dim),
                                             nn.SiLU())
        self.dimen_redu = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(hidden_dim),
            # nn.GroupNorm(4, hid_chans),
            nn.SiLU(),

            nn.Conv2d(hidden_dim, hidden_dim, 1, 1, 0, bias=True),
            nn.BatchNorm2d(hidden_dim),
            # nn.GroupNorm(4, hid_chans),
            # nn.SiLU(),
            )
    def forward(self, x):
        if self.group_norm:
            x = self.patch_embedding(x) # B,C,H,W
        else:
            x = self.dimen_redu(x)
        return x
    

def seq_generate(x, max_adj=4, if_pos_embed=True):
        B, C, H, W = x.shape
        center_root = math.floor(H/2) * W + math.floor(W/2)

        bfs = _BFS.apply
        mst = MinimumSpanningTree("Cosine", torch.exp)
        tree = mst(x)
        sorted_index,sorted_parent,sorted_child = bfs(tree,max_adj,center_root)

        if if_pos_embed:
            pos_embed = get_2d_sincos_pos_embed(C, H)
            pos_embed = pos_embed.view(H, W, C)
            pos_embed = pos_embed.unsqueeze(0).expand(B, -1, -1, -1) # (B,H,W,C)
            x = x.permute(0, 2, 3, 1)
            x = x + pos_embed
            x = rearrange(x, 'b h w c -> b (h w) c')
        else:
            x = x.permute(0, 2, 3, 1)
            x = rearrange(x, 'b h w c -> b (h w) c')

        x_sorted = x.gather(1, sorted_index.unsqueeze(-1).expand(B, H*W, C))

        return sorted_index, x_sorted


def seq_resize(x, sorted_index, patch_size):
    B,L,C = x.shape
    inverse_index = torch.argsort(sorted_index, dim=1)
    inverse_index_expanded = inverse_index.unsqueeze(-1).expand(-1, -1, C)
    restored_x = torch.gather(x, dim=1, index=inverse_index_expanded)

    resize_x = rearrange(restored_x, 'b (h w) c -> b h w c', h=patch_size,w=patch_size)

    return resize_x


class DownsampleLayer(nn.Module):
    r""" Downsample layer
    Args:
        channels (int): number of input channels
        norm_layer (str): normalization layer
    """

    def __init__(self, channels, patch_size):
        super().__init__()
        self.conv = nn.Conv2d(channels,
                              channels,
                              kernel_size=3,
                              stride=2,
                              padding=1,
                              bias=False)
        self.norm = nn.LayerNorm(channels)
        self.patch_size = patch_size

    def forward(self, x, sorted_index):
        x = seq_resize(x, sorted_index, self.patch_size) # B,H,W,C
        x = self.conv(x.permute(0,3,1,2)) # B,C,H,W
        x = self.norm(x.permute(0,2,3,1)) # B,H,W,C

        return x


class Spa_DownsampleLayer(nn.Module):
    def __init__(self, embed_dim, num_heads,patch_size, if_resize=True):
        super().__init__()
        self.cross_atten = CrossAttention(embed_dim, num_heads)
        self.if_resize = if_resize
        self.patch_size = patch_size
    
    def forward(self, x, sorted_index):
        if self.if_resize:
            x = seq_resize(x, sorted_index, self.patch_size)
            sorted_index,x = seq_generate(x.permute(0, 3, 1, 2)) 
        
        l = ((self.patch_size+1) // 2) ** 2 
        x = self.cross_atten(x[:, :l, :], x, x)

        return x,sorted_index
        
class SSD_Block(nn.Module):
    def __init__(self, channels, seq_len, drop_path=0.):
        super(SSD_Block, self).__init__()
        self.spa_dim = channels
        self.spe_dim = seq_len
        self.spa_mamba = Mamba2(
            # This module uses roughly 3 * expand * d_model^2 parameters
            d_model=self.spa_dim, # Model dimension d_model
            d_state=64,  # SSM state expansion factor
            d_conv=4,    # Local convolution width
            expand=2,    # Block expansion factor
        )
        self.spe_mamba = Mamba2(
            # This module uses roughly 3 * expand * d_model^2 parameters
            d_model=self.spe_dim, # Model dimension d_model
            d_state=64,  # SSM state expansion factor
            d_conv=4,    # Local convolution width
            expand=2,    # Block expansion factor
        )
        self.spa_norm = nn.LayerNorm(channels)
        self.spe_norm = nn.LayerNorm(seq_len)
        self.drop_path = DropPath(drop_path) if drop_path > 0. \
            else nn.Identity()

    def forward(self, x):
        x_spa = self.spa_norm(x) # (B,L,C)
        x_spa = self.spa_mamba(x_spa)
        x_spa = x_spa + self.drop_path(x)

        x_spe = x_spa.permute(0, 2, 1) # (B,C,L)

        x_spe = self.spe_norm(x_spe)
        x_spe = self.drop_path(self.spe_mamba(x_spe))
        x_spe = x_spe.permute(0, 2, 1)
        x_out = x_spe + x_spa

        return x_out

class Basic_Block_v1(nn.Module):
    def __init__(self, 
                 channels, 
                 depth, 
                 patch_size, 
                 num_heads, 
                 drop_path=0., 
                 spa_query_len=5,
                 downsample=True
                 ):
        super(Basic_Block_v1, self).__init__()
        self.seq_len = patch_size ** 2
        self.spa_query_len = spa_query_len
        self.ssd_blocks = nn.ModuleList([
            SSD_Block(
                channels=channels, 
                seq_len=patch_size**2,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                )for i in range(depth)])
        self.spa_attention = Spa_Attention(channels, num_heads)
        self.spe_attention = Spe_Attention(channels, self.seq_len)
        self.center_prj = nn.Conv1d(in_channels=channels, 
                                    out_channels=channels, 
                                    kernel_size=spa_query_len, 
                                    stride=1, 
                                    padding=0)
        self.norm = nn.LayerNorm(channels)
        self.downsample = DownsampleLayer(
            channels=channels, 
            patch_size=patch_size) if downsample else None
    

    def forward(self, x):
        B, C, H, W = x.shape

        sorted_index,x = seq_generate(x) # (B,L,C)
        for i, blocks in enumerate(self.ssd_blocks):
            x = blocks(x)
        x = self.norm(x)

        center_embed = self.center_prj(x[:,:self.spa_query_len,:].permute(0,2,1)) # B,C,1
        x = self.spa_attention(center_embed.permute(0,2,1), x, x)
        x = self.spe_attention(x)

        if self.downsample is not None:
            x = self.downsample(x,sorted_index)

        return x
    

class Basic_Block_v2(nn.Module):
    def __init__(self, channels, patch_size, depth, num_heads, if_resize, drop_path=0.):
        super(Basic_Block_v1, self).__init__()
        self.seq_len = patch_size ** 2
        self.ssd_blocks = nn.ModuleList([
            SSD_Block(
                channels=channels, 
                seq_len=patch_size**2,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                )for i in range(depth)])
        self.spa_attention = Spa_DownsampleLayer(channels,
                                                 num_heads, 
                                                 patch_size,
                                                 if_resize=if_resize)
        self.spe_attention = Spe_Attention(channels, self.seq_len)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x, sorted_index):
        B, L, C = x.shape
        for i, blocks in enumerate(self.ssd_blocks):
            x = blocks(x)
        x = self.norm(x)

        x,sorted_i = self.spa_attention(x,sorted_index)
        x = self.spe_attention(x)

        return x, sorted_i
    

class Main_Model(nn.Module):
    def __init__(self, 
                 in_channels,
                 embed_dim=64,
                 num_classes=10,
                 patch_size=17,
                 group_num=4,
                 group_norm=True,
                 basic_version='v1',
                 depths=[2, 4, 2],
                 drop_path_rate=0.2,
                 drop_path_type='linear',
                 num_head=8,
                 spa_query_len=5,
                 head_typ='AP',
                 v2_if_resize = True,
                 ):
        super().__init__()
        self.depths = depths
        self.num_levels = len(depths)
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.version = basic_version
        self.patch_embed = PatchEmbed(in_channels,
                                      embed_dim,
                                      group_num,
                                      group_norm)
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        if drop_path_type == 'uniform':
            for i in range(len(dpr)):
                dpr[i] = drop_path_rate

        self.levels = nn.ModuleList()
        if basic_version == 'v1':
            for i in range(self.num_levels):
                level = Basic_Block_v1(
                    channels=embed_dim,
                    depth=depths[i],
                    patch_size=self.patch_size,
                    num_heads=num_head,
                    drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                    spa_query_len=spa_query_len-2*i,
                    downsample=(i < self.num_levels - 1)
                    )
                self.patch_size = math.floor((self.patch_size-1)/2)+1
                self.levels.append(level)
        if basic_version == 'v2':
            for i in range(self.num_levels):
                level = Basic_Block_v2(
                    channels=embed_dim,
                    patch_size=self.patch_size,
                    depth=depths[i],
                    num_heads=num_head,
                    if_resize=v2_if_resize,
                    drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])]
                )
                self.patch_size = math.floor((self.patch_size-1)/2)+1
                self.levels.append(level)
        
        self.head = Head(embed_dim,
                         num_classes,
                         sequence_length=self.patch_size**2,
                         head=head_typ)
        
    def forward(self, x):
        x = self.patch_embed(x)

        if self.version == 'v1':
            for level in self.levels:
                x = level(x)
        if self.version == 'v2':
            sorted_index, x = seq_generate(x)
            for level in self.levels:
                x, sorted_index = level(x, sorted_index)
        
        x = self.head(x)

        return x
    




        
    











    

    





