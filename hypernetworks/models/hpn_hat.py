import torch.nn as nn
import torch

def convert_arch_params_to_dict(arch_params):
    arch_params_dict = {}
    arch_params_dict["encoder-embed-dim"] = arch_params[0]
    arch_params_dict["decoder-embed-dim"] = arch_params[1]
    arch_params_dict["encoder-layer-num"] = arch_params[2]
    arch_params_dict["decoder-layer-num"] = arch_params[3]
    arch_params_dict["encoder-ffn-embed-dim"] = arch_params[4]
    arch_params_dict["decoder-ffn-embed-dim"] = arch_params[5]
    arch_params_dict["encoder-self-attention-heads"] = arch_params[6]
    arch_params_dict["decoder-self-attention-heads"] = arch_params[7]
    arch_params_dict["decoder-ende-attention-heads"] = arch_params[8]
    arch_params_dict["decoder-arbitrary-ende-attn"] = arch_params[9]
    return arch_params_dict

class HyperNetwork(nn.Module):
    def __init__(self, search_space):
        super(HyperNetwork, self).__init__()
        self.num_choices_encoder_embed = len(search_space["encoder-embed-choice"])
        self.num_choices_decoder_embed = len(search_space["decoder-embed-choice"])
        self.num_choices_encoder_layer = len(search_space["encoder-layer-num-choice"])
        self.num_choices_decoder_layer = len(search_space["decoder-layer-num-choice"])
        self.num_choices_encoder_ffn_embed_dim = len(search_space["encoder-ffn-embed-dim-choice"])
        self.num_choices_decoder_ffn_embed_dim = len(search_space["decoder-ffn-embed-dim-choice"])
        self.num_choices_encoder_attention_heads = len(search_space["encoder-self-attention-heads-choice"])
        self.num_choices_decoder_attention_heads = len(search_space["decoder-self-attention-heads-choice"])
        self.num_choices_decoder_ende_attention_heads = len(search_space["decoder-ende-attention-heads-choice"])
        self.num_choices_decoder_arbitrary_ende_attn = len(search_space["decoder-arbitrary-ende-attn-choice"])
        self.max_enc_layers = max(search_space["encoder-layer-num-choice"])
        self.max_dec_layers = max(search_space["decoder-layer-num-choice"])

        self.embedding_encoder_embed = nn.Embedding(101, self.num_choices_encoder_embed).cuda()
        self.embedding_decoder_embed = nn.Embedding(101, self.num_choices_decoder_embed).cuda()
        self.embedding_encoder_layer = nn.Embedding(101, self.num_choices_encoder_layer).cuda()
        self.embedding_decoder_layer = nn.Embedding(101, self.num_choices_decoder_layer).cuda()
        self.embedding_encoder_ffn_embed_dim = nn.Embedding(101, self.num_choices_encoder_ffn_embed_dim*self.max_enc_layers).cuda()
        self.embedding_decoder_ffn_embed_dim = nn.Embedding(101, self.num_choices_decoder_ffn_embed_dim*self.max_dec_layers).cuda()
        self.embedding_encoder_attention_heads = nn.Embedding(101, self.num_choices_encoder_attention_heads*self.max_enc_layers).cuda()
        self.embedding_decoder_attention_heads = nn.Embedding(101, self.num_choices_decoder_attention_heads*self.max_dec_layers).cuda()
        self.embedding_decoder_ende_attention_heads = nn.Embedding(101, self.num_choices_decoder_ende_attention_heads*self.max_dec_layers).cuda()
        self.embedding_decoder_arbitrary_ende_attn = nn.Embedding(101, self.num_choices_decoder_arbitrary_ende_attn*self.max_dec_layers).cuda()


    def forward(self, x):
        id = int(x[0][0]*100)
        #print(id)
        encoder_embed = self.embedding_encoder_embed(torch.tensor([id]).to(x.device))
        decoder_embed = self.embedding_decoder_embed(torch.tensor([id]).to(x.device))
        encoder_layer = self.embedding_encoder_layer(torch.tensor([id]).to(x.device))
        decoder_layer = self.embedding_decoder_layer(torch.tensor([id]).to(x.device))
        encoder_ffn_embed_dim = self.embedding_encoder_ffn_embed_dim(torch.tensor([id]).to(x.device)).reshape(self.max_dec_layers, self.num_choices_encoder_ffn_embed_dim)
        decoder_ffn_embed_dim = self.embedding_decoder_ffn_embed_dim(torch.tensor([id]).to(x.device)).reshape(self.max_dec_layers, self.num_choices_decoder_ffn_embed_dim)
        encoder_attention_heads = self.embedding_encoder_attention_heads(torch.tensor([id]).to(x.device)).reshape(self.max_dec_layers, self.num_choices_encoder_attention_heads)
        decoder_attention_heads = self.embedding_decoder_attention_heads(torch.tensor([id]).to(x.device)).reshape(self.max_dec_layers, self.num_choices_decoder_attention_heads)
        decoder_ende_attention_heads = self.embedding_decoder_ende_attention_heads(torch.tensor([id]).to(x.device)).reshape(self.max_dec_layers, self.num_choices_decoder_ende_attention_heads)
        decoder_arbitrary_ende_attn = self.embedding_decoder_arbitrary_ende_attn(torch.tensor([id]).to(x.device)).reshape(self.max_dec_layers, self.num_choices_decoder_arbitrary_ende_attn)
        return encoder_embed, decoder_embed, encoder_layer, decoder_layer, encoder_ffn_embed_dim, decoder_ffn_embed_dim, encoder_attention_heads, decoder_attention_heads, decoder_ende_attention_heads, decoder_arbitrary_ende_attn
    
    
class MetaHyperNetwork(nn.Module):
    def __init__(self, choices, num_random_devices=50, hw_embed_dim=10, hpn_type=HyperNetwork):
        super(MetaHyperNetwork, self).__init__()
        self.hardware_embedding = nn.Embedding(num_random_devices, hw_embed_dim)
        self.hpn_factory = [hpn_type(choices) for i in range(num_random_devices)]
        self.hpn_factory = nn.ModuleList(self.hpn_factory)

        self.hw_embed_dim = hw_embed_dim

    def forward(self, x, hw):
        hw = hw.float()
        similarity = torch.squeeze(torch.matmul(hw, self.hardware_embedding.weight.to(x.device).transpose(0,1)))/torch.sqrt(torch.tensor(self.hw_embed_dim*1.0).to(x.device))
        # compute softmax
        similarity = torch.softmax(similarity, dim=-1)
        out_encoder_embed = torch.zeros([self.hpn_factory[0].num_choices_encoder_embed]).to(x.device)
        out_decoder_embed = torch.zeros([self.hpn_factory[0].num_choices_decoder_embed]).to(x.device)
        out_encoder_layer = torch.zeros([self.hpn_factory[0].num_choices_encoder_layer]).to(x.device)
        out_decoder_layer = torch.zeros([self.hpn_factory[0].num_choices_decoder_layer]).to(x.device)
        out_encoder_ffn_embed_dim = torch.zeros([self.hpn_factory[0].max_enc_layers,self.hpn_factory[0].num_choices_encoder_ffn_embed_dim]).to(x.device)
        out_decoder_ffn_embed_dim = torch.zeros([self.hpn_factory[0].max_dec_layers,self.hpn_factory[0].num_choices_decoder_ffn_embed_dim]).to(x.device)
        out_encoder_attention_heads = torch.zeros([self.hpn_factory[0].max_enc_layers,self.hpn_factory[0].num_choices_encoder_attention_heads]).to(x.device)
        out_decoder_attention_heads = torch.zeros([self.hpn_factory[0].max_dec_layers,self.hpn_factory[0].num_choices_decoder_attention_heads]).to(x.device)
        out_decoder_ende_attention_heads = torch.zeros([self.hpn_factory[0].max_dec_layers,self.hpn_factory[0].num_choices_decoder_ende_attention_heads]).to(x.device)
        out_decoder_arbitrary_ende_attn = torch.zeros([self.hpn_factory[0].max_dec_layers,self.hpn_factory[0].num_choices_decoder_arbitrary_ende_attn]).to(x.device)
        for i in range(similarity.shape[0]):
            hpn_out_encoder_embed, hpn_out_decoder_embed, hpn_out_encoder_layer, hpn_out_decoder_layer, hpn_out_encoder_ffn_embed_dim, hpn_out_decoder_ffn_embed_dim, hpn_out_encoder_attention_heads, hpn_out_decoder_attention_heads, hpn_out_decoder_ende_attention_heads, hpn_out_decoder_arbitrary_ende_attn = self.hpn_factory[i](x)
            out_encoder_embed+= similarity[i]*torch.squeeze(hpn_out_encoder_embed)
            out_decoder_embed+= similarity[i]*torch.squeeze(hpn_out_decoder_embed)
            out_encoder_layer+= similarity[i]*torch.squeeze(hpn_out_encoder_layer)
            out_decoder_layer+= similarity[i]*torch.squeeze(hpn_out_decoder_layer)
            out_encoder_ffn_embed_dim+= similarity[i]*torch.squeeze(hpn_out_encoder_ffn_embed_dim)
            out_decoder_ffn_embed_dim+= similarity[i]*torch.squeeze(hpn_out_decoder_ffn_embed_dim)
            out_encoder_attention_heads+= similarity[i]*torch.squeeze(hpn_out_encoder_attention_heads)
            out_decoder_attention_heads+= similarity[i]*torch.squeeze(hpn_out_decoder_attention_heads)
            out_decoder_ende_attention_heads+= similarity[i]*torch.squeeze(hpn_out_decoder_ende_attention_heads)
            out_decoder_arbitrary_ende_attn+= similarity[i]*torch.squeeze(hpn_out_decoder_arbitrary_ende_attn)
    
        return out_encoder_embed, out_decoder_embed, out_encoder_layer, out_decoder_layer, out_encoder_ffn_embed_dim, out_decoder_ffn_embed_dim, out_encoder_attention_heads, out_decoder_attention_heads, out_decoder_ende_attention_heads, out_decoder_arbitrary_ende_attn