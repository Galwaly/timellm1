# TimeLLM.py
from math import sqrt
import torch
import torch.nn as nn
from transformers import BertConfig, BertModel, BertTokenizer
from layers.Embed import PatchEmbedding
from layers.StandardNorm import Normalize
import transformers
transformers.logging.set_verbosity_error()


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        return self.dropout(self.linear(self.flatten(x)))


class ReprogrammingLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_keys=None, d_llm=768, attention_dropout=0.1):
        super().__init__()
        d_keys = d_keys or (d_model // n_heads)
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.value_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, d_llm)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, target_embedding, source_embedding, value_embedding):
        B, L, _ = target_embedding.shape
        S, _ = source_embedding.shape
        H = self.n_heads
        target_embedding = self.query_projection(target_embedding).view(B, L, H, -1)
        source_embedding = self.key_projection(source_embedding).view(S, H, -1)
        value_embedding = self.value_projection(value_embedding).view(S, H, -1)
        out = self.reprogramming(target_embedding, source_embedding, value_embedding)
        out = out.reshape(B, L, -1)
        return self.out_projection(out)

    def reprogramming(self, target_embedding, source_embedding, value_embedding):
        B, L, H, E = target_embedding.shape
        scale = 1.0 / sqrt(E)
        scores = torch.einsum("blhe,she->bhls", target_embedding, source_embedding)
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        return torch.einsum("bhls,she->blhe", A, value_embedding)


class Model(nn.Module):
    def __init__(self, configs, patch_len=16, stride=8):
        super().__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.d_ff = configs.d_ff
        self.top_k = 5
        self.d_llm = 768  # BERT 原生维度
        self.patch_len = configs.patch_len
        self.stride = configs.stride
        configs.llm_model = 'BERT'

        # 1. 加载 BERT（可本地可在线）
        self.bert_config = BertConfig.from_pretrained('/tmp/pycharm_project_254/BERT')
        self.bert_config.num_hidden_layers = configs.llm_layers
        self.llm_model = BertModel.from_pretrained('/tmp/pycharm_project_254/BERT', config=self.bert_config)
        self.tokenizer = BertTokenizer.from_pretrained('/tmp/pycharm_project_254/BERT')

        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.llm_model.resize_token_embeddings(len(self.tokenizer))

        # 2. 放开训练（关键）
        # for p in self.llm_model.parameters():
        #     p.requires_grad = False

        self.description = (configs.content if configs.prompt_domain else
                            'The Electricity Transformer Temperature (ETT) is a crucial indicator '
                            'in the electric power long-term deployment.')

        self.dropout = nn.Dropout(configs.dropout)
        self.patch_embedding = PatchEmbedding(configs.d_model, self.patch_len, self.stride, configs.dropout)

        self.word_embeddings = self.llm_model.get_input_embeddings().weight
        self.vocab_size = self.word_embeddings.shape[0]
        self.num_tokens = 1000
        self.mapping_layer = nn.Linear(self.vocab_size, self.num_tokens)

        self.reprogramming_layer = ReprogrammingLayer(configs.d_model, configs.n_heads, self.d_ff, self.d_llm)

        self.patch_nums = int((configs.seq_len - self.patch_len) / self.stride + 2)
        self.head_nf = self.d_ff * self.patch_nums

        if self.task_name in {'long_term_forecast', 'short_term_forecast'}:
            self.output_projection = FlattenHead(configs.enc_in, self.head_nf, self.pred_len, configs.dropout)
        else:
            raise NotImplementedError

        self.normalize_layers = Normalize(configs.enc_in, affine=False)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in {'long_term_forecast', 'short_term_forecast'}:
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]
        return None

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        x_enc = self.normalize_layers(x_enc, 'norm')
        B, T, N = x_enc.size()
        x_enc = x_enc.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)

        min_values = torch.min(x_enc, dim=1)[0]
        max_values = torch.max(x_enc, dim=1)[0]
        medians = torch.median(x_enc, dim=1).values
        lags = self.calcute_lags(x_enc)
        trends = x_enc.diff(dim=1).sum(dim=1)

        prompt = []
        for b in range(x_enc.shape[0]):
            min_str = str(min_values[b].tolist()[0])
            max_str = str(max_values[b].tolist()[0])
            med_str = str(medians[b].tolist()[0])
            lag_str = str(lags[b].tolist())
            prompt.append(
                f"<|start_prompt|>Dataset description: {self.description}"
                f"Task description: forecast the next {self.pred_len} steps given "
                f"the previous {self.seq_len} steps information; "
                f"Input statistics: min value {min_str}, max value {max_str}, "
                f"median value {med_str}, "
                f"trend is {'upward' if trends[b] > 0 else 'downward'}, "
                f"top 5 lags are: {lag_str}<|<end_prompt>|>"
            )

        x_enc = x_enc.reshape(B, N, T).permute(0, 2, 1).contiguous()

        # prompt -> 768
        prompt_ids = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids.to(x_enc.device)
        prompt_embeddings = self.llm_model.get_input_embeddings()(prompt_ids)  # (B, Lp, 768)

        source_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)  # (1000, 768)

        x_enc = x_enc.permute(0, 2, 1).contiguous()
        enc_out, n_vars = self.patch_embedding(x_enc.to(torch.bfloat16))  # (B*vars, patch, d_model)
        enc_out = self.reprogramming_layer(enc_out, source_embeddings, source_embeddings)  # -> (B*vars, patch, 768)

        llama_enc_out = torch.cat([prompt_embeddings, enc_out], dim=1)  # 768 维拼接
        dec_out = self.llm_model(inputs_embeds=llama_enc_out).last_hidden_state  # (B, L, 768)
        dec_out = dec_out[:, :, :self.d_ff]  # 取前 d_ff 维特征

        dec_out = dec_out.reshape(-1, n_vars, dec_out.size(-2), dec_out.size(-1))
        dec_out = dec_out.permute(0, 1, 3, 2).contiguous()
        dec_out = self.output_projection(dec_out[:, :, :, -self.patch_nums:])
        dec_out = dec_out.permute(0, 2, 1).contiguous()

        return self.normalize_layers(dec_out, 'denorm')

    def calcute_lags(self, x_enc):
        x = x_enc.permute(0, 2, 1).contiguous()
        q_fft = torch.fft.rfft(x, dim=-1)
        k_fft = torch.fft.rfft(x, dim=-1)
        corr = torch.fft.irfft(q_fft * torch.conj(k_fft), dim=-1)
        mean_c = torch.mean(corr, dim=1)
        _, lags = torch.topk(mean_c, self.top_k, dim=-1)
        return lags