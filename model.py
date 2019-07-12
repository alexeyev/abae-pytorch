# -*- coding: utf-8 -*-
import torch
from torch.nn import init
from torch.nn.parameter import Parameter


class SelfAttention(torch.nn.Module):
    def __init__(self, wv_dim, maxlen):
        super(SelfAttention, self).__init__()
        self.wv_dim = wv_dim
        self.maxlen = maxlen
        self.M = Parameter(torch.Tensor(wv_dim, wv_dim))
        init.kaiming_uniform(self.M.data)
        self.attention_softmax = torch.nn.Softmax()
        # self.reset_parameters()

    # def reset_parameters(self):
    #     init.kaiming_uniform_(self.weight, a=math.sqrt(5))
    #     if self.bias is not None:
    #         fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
    #         bound = 1 / math.sqrt(fan_in)
    #         init.uniform_(self.bias, -bound, bound)

    def forward(self, input_embeddings):
        # (b, wv, 1)
        mean_embedding = torch.mean(input_embeddings, (1,)).unsqueeze(2)

        # (wv, wv) x (b, wv, 1) -> (b, wv, 1)
        product_1 = torch.matmul(self.M, mean_embedding)

        # (b, maxlen, wv) x (b, wv, 1) -> (b, maxlen, 1)
        product_2 = torch.matmul(input_embeddings, product_1).squeeze(2)

        results = self.attention_softmax(product_2)

        return results

    def extra_repr(self):
        return 'wv_dim={}, maxlen={}'.format(self.wv_dim, self.maxlen)


class ABAE(torch.nn.Module):
    def __init__(self, wv_dim=200, asp_count=30, ortho_reg=0.1, maxlen=201,
                 negative_samples=5, init_aspects_matrix=None):
        super(ABAE, self).__init__()
        self.wv_dim = wv_dim
        self.asp_count = asp_count
        self.ortho = ortho_reg
        self.maxlen = maxlen
        self.negative_samples = negative_samples

        self.m = torch.randn(wv_dim, wv_dim)
        self.attention = SelfAttention(wv_dim, maxlen)
        self.linear_transform = torch.nn.Linear(self.wv_dim, self.asp_count)
        self.softmax_aspects = torch.nn.Softmax()
        self.aspects_embeddings = Parameter(torch.Tensor(wv_dim, asp_count))

        if init_aspects_matrix is None:
            torch.nn.init.xavier_uniform(self.aspects_embeddings)
        else:
            self.aspects_embeddings.data = torch.from_numpy(init_aspects_matrix.T)

        self.loss = torch.nn.TripletMarginLoss

    def forward(self, text_embeddings, negative_samples_texts):

        averaged_negative_samples = torch.mean(negative_samples_texts, dim=2)

        attention_weights, aspects_importances, weighted_text_emb = self.get_aspects_importances(text_embeddings)
        recovered_emb = torch.matmul(self.aspects_embeddings, aspects_importances.unsqueeze(2)).squeeze()

        reconstruction_triplet_loss = self._reconstruction_loss(weighted_text_emb,
                                                                recovered_emb,
                                                                averaged_negative_samples)

        max_margin = torch.max(reconstruction_triplet_loss, torch.zeros_like(reconstruction_triplet_loss))

        return self.ortho * self.ortho_regularizer() + max_margin

    def _reconstruction_loss(self, text_emb, recovered_emb, averaged_negative_emb):
        positive_dot_products = torch.matmul(text_emb.unsqueeze(1), recovered_emb.unsqueeze(2)).squeeze()
        negative_dot_products = torch.matmul(averaged_negative_emb, recovered_emb.unsqueeze(2)).squeeze()

        sum_negative_dot = torch.sum(negative_dot_products, dim=1)

        reconstruction_triplet_loss = 1 - positive_dot_products + sum_negative_dot

        return reconstruction_triplet_loss

    def ortho_regularizer(self):
        return torch.norm(
            torch.matmul(self.aspects_embeddings.t(), self.aspects_embeddings) \
            - torch.eye(self.asp_count))

    def get_aspects_importances(self, text_embeddings):
        attention_weights = self.attention(text_embeddings)
        # print("attention:", attention_weights)
        weighted_text_emb = torch.matmul(attention_weights.unsqueeze(1), text_embeddings).squeeze()
        raw_importances = self.linear_transform(weighted_text_emb)
        aspects_importances = self.softmax_aspects(raw_importances)

        return attention_weights, aspects_importances, weighted_text_emb

    def get_aspect_words(self, w2v_model, topn=15):
        words = []

        for row in self.aspects_embeddings.t().detach().numpy():
            words.append([w for w, dist in w2v_model.similar_by_vector(row)[:topn]])

        return words
