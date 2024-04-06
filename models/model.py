import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layers.RevIN import RevIN
from models.layers.GCN import GCN
from models.layers.Encoder import channel_shuffle,AttentionBlock

class JointConv(nn.Module):
    def __init__(self, in_channels, partial):
        super(JointConv, self).__init__()
        
        self.in_channels = in_channels
        k_size = 3
        self.partial = partial
        pc = int(in_channels * (1 / partial))


        self.T_conv = nn.Conv2d(pc, pc, kernel_size=(1, k_size), padding=(0,1))
        self.S_conv = nn.Conv2d(pc, pc, kernel_size=(k_size,1), padding=(1,0))
        self.ST_conv = nn.Conv2d(pc, pc, kernel_size=(k_size,k_size), padding=(1,1))
        self.split_indexes = (in_channels - 3 * pc, pc, pc, pc)
        self.dropout_layer = nn.Dropout(0.2)

    def forward(self, x):
        b,c,n,t = x.shape
        x_id, x_st, x_s, x_t = torch.split(x, self.split_indexes, dim=1)
        output = torch.cat(
            (x_id, self.ST_conv(x_st), self.S_conv(x_s), self.T_conv(x_t)), 
            dim=1,
        )

        output = channel_shuffle(output, self.partial)

        return output

class JointAttention(nn.Module):
    def __init__(self, in_channels, num_heads, partial):
        super(JointAttention, self).__init__()  

        self.in_channels = in_channels
        self.partial = partial
        pc = int(in_channels * (1 / partial))
        self.channels = pc
        self.T_attention = AttentionBlock(dim=pc, num_heads=num_heads)
        self.S_attention = AttentionBlock(dim=pc, num_heads=num_heads)
        self.ST_attention = AttentionBlock(dim=pc, num_heads=num_heads)
        self.split_indexes = (in_channels - 3 * pc, pc, pc, pc)

        self.dropout_layer = nn.Dropout(0.2)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels, in_channels),
        )
        self.ln1 = nn.LayerNorm(in_channels)
        self.ln2 = nn.LayerNorm(in_channels)
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)

    def forward(self, x):

        b,n,t,c = x.shape

        x_id, x_st, x_s, x_t = torch.split(x, self.split_indexes, dim=3)
        t_x = x_t.reshape(b*n,t,self.channels)
        s_x = x_s.permute(0,2,1,3).reshape(b*t,n,self.channels)
        st_x = x_st.reshape(b,-1,self.channels)
        
        id_x = x_id
        t_x = self.T_attention(t_x).reshape(b,n,t,self.channels)
        s_x = self.S_attention(s_x).reshape(b,t,n,self.channels).permute(0,2,1,3)
        st_x = self.ST_attention(st_x).reshape(b,n,t,self.channels)

        output = torch.cat((id_x,t_x,s_x,st_x), dim=3).permute(0,3,1,2)

        output = channel_shuffle(output, self.partial)
        return output
    

class ASTCL(nn.Module):
    def __init__(self, channels, heads, depth, partial, num_features, num_timesteps_input, num_timesteps_output, node_num,
                  dropout=0.5, revin=True, graph=True):
        super(ASTCL, self).__init__()

        self.depth = depth
        self.partial = partial

        self.start_conv = nn.ModuleList()
        self.joint_conv = nn.ModuleList()
        self.joint_attention = nn.ModuleList()
        self.residual = nn.ModuleList()
        self.revin_layer = RevIN(num_features, affine=True, subtract_last=False)

        self.revin = revin
        self.graph = graph

        for i in range(depth):
            self.start_conv.append(nn.Conv2d(in_channels=channels[i],
                                             out_channels=channels[i+1],
                                             kernel_size = (1,1)))
            self.joint_conv.append(JointConv(channels[i+1], self.partial))
            self.joint_attention.append(JointAttention(channels[i+1], heads[i], self.partial))

        self.end_conv = nn.Conv2d(in_channels=channels[-1],out_channels=num_features,kernel_size=3,padding=1)

        self.dropout_layer = nn.Dropout(p=dropout)

        if graph:
            self.gcn = nn.ModuleList()
            for i in range(depth):
                self.gcn.append(GCN(c_in=channels[0],c_out=channels[i+1],dropout=0.2,support_len=1,order=2))


        
    def forward(self, input, adj):

        input = input.float()
        if self.revin: 
            input = self.revin_layer(input, 'norm')
        input = input.permute(0, 3, 1, 2).contiguous()
        b,c,n,l = input.shape
        #(32,3,15,12)
        x = input
        
        if self.graph:
            x_gcn = []
            for i in range(self.depth):
                x_gcn.append(self.gcn[i](x,adj))
            
            for i in range(self.depth):

                x = F.leaky_relu(self.start_conv[i](x), 0.2)
                shortcut = x
                x = F.leaky_relu(self.joint_attention[i](x.permute(0,2,3,1)), 0.2)
                x = F.leaky_relu(self.joint_conv[i](x), 0.2)

                x = x + x_gcn[i] + shortcut
        else:
            for i in range(self.depth):
                # b,c,n,l
                x = F.leaky_relu(self.start_conv[i](x), 0.2)
                shortcut = x
                x = F.leaky_relu(self.joint_attention[i](x.permute(0,2,3,1)), 0.2)
                x = F.leaky_relu(self.joint_conv[i](x), 0.2)
                # b,c,n,l
                x = shortcut + x

        x = self.dropout_layer(x)
        output = self.end_conv(x)

        if self.revin: 
            output = torch.squeeze(self.revin_layer(output.permute(0,2,3,1), 'denorm')[:,0,:,0])
        else:
            output = torch.squeeze(output.permute(0,2,3,1)[:,0,:,0])
        
        return output
    
    def initialize(self):
        for m in self.modules():
            for p in m.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
                else:
                    nn.init.uniform_(p)

