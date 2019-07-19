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
    """
        The model described in the paper ``An Unsupervised Neural Attention Model for Aspect Extraction''
        by He, Ruidan and  Lee, Wee Sun  and  Ng, Hwee Tou  and  Dahlmeier, Daniel, ACL2017
        https://aclweb.org/anthology/papers/P/P17/P17-1036/

    """
    def __init__(self, wv_dim=200, asp_count=30, ortho_reg=0.1, maxlen=201, init_aspects_matrix=None):
        """
        Initializing the model

        :param wv_dim: word vector size
        :param asp_count: number of aspects
        :param ortho_reg: coefficient for tuning the ortho-regularizer's influence
        :param maxlen: sentence max length taken into account
        :param init_aspects_matrix: None or init. matrix for aspects
        """
        super(ABAE, self).__init__()
        self.wv_dim = wv_dim
        self.asp_count = asp_count
        self.ortho = ortho_reg
        self.maxlen = maxlen

        self.attention = SelfAttention(wv_dim, maxlen)
        self.linear_transform = torch.nn.Linear(self.wv_dim, self.asp_count)
        self.softmax_aspects = torch.nn.Softmax()
        self.aspects_embeddings = Parameter(torch.Tensor(wv_dim, asp_count))

        if init_aspects_matrix is None:
            torch.nn.init.xavier_uniform(self.aspects_embeddings)
        else:
            self.aspects_embeddings.data = torch.from_numpy(init_aspects_matrix.T)

    def get_aspects_importances(self, text_embeddings):
        """
            Takes embeddings of a sentence as input, returns attention weights
        """

        attention_weights = self.attention(text_embeddings)
        weighted_text_emb = torch.matmul(attention_weights.unsqueeze(1), text_embeddings).squeeze()
        raw_importances = self.linear_transform(weighted_text_emb)
        aspects_importances = self.softmax_aspects(raw_importances)

        return attention_weights, aspects_importances, weighted_text_emb

    def forward(self, text_embeddings, negative_samples_texts):

        averaged_negative_samples = torch.mean(negative_samples_texts, dim=2)

        attention_weights, aspects_importances, weighted_text_emb = self.get_aspects_importances(text_embeddings)
        recovered_emb = torch.matmul(self.aspects_embeddings, aspects_importances.unsqueeze(2)).squeeze()

        reconstruction_triplet_loss = ABAE._reconstruction_loss(weighted_text_emb,
                                                                recovered_emb,
                                                                averaged_negative_samples)

        max_margin = torch.max(reconstruction_triplet_loss, torch.zeros_like(reconstruction_triplet_loss))

        return self.ortho * self._ortho_regularizer() + max_margin

    @staticmethod
    def _reconstruction_loss(text_emb, recovered_emb, averaged_negative_emb):

        positive_dot_products = torch.matmul(text_emb.unsqueeze(1), recovered_emb.unsqueeze(2)).squeeze()
        negative_dot_products = torch.matmul(averaged_negative_emb, recovered_emb.unsqueeze(2)).squeeze()
        sum_negative_dot = torch.sum(negative_dot_products, dim=1)
        reconstruction_triplet_loss = 1 - positive_dot_products + sum_negative_dot

        return reconstruction_triplet_loss

    def _ortho_regularizer(self):
        return torch.norm(
            torch.matmul(self.aspects_embeddings.t(), self.aspects_embeddings) \
            - torch.eye(self.asp_count))

    def get_aspect_words(self, w2v_model, topn=15):
        words = []

        for row in self.aspects_embeddings.t().detach().numpy():
            words.append([w for w, dist in w2v_model.similar_by_vector(row)[:topn]])

        return words
