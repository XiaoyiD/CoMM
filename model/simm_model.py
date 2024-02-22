# -*- coding: UTF-8 -*-
import torch.nn as nn
import torch
import random

from model.utilities.common_loss import My_KL_loss
from model.utilities.common_loss import My_logit_ML_loss

import torch.nn.functional as F


class SIMM_Model(nn.Module):

    def __init__(self, view_blocks, comm_feature_num, label_num, model_args=None, projection_dim=64):
        super(SIMM_Model, self).__init__()
        self.view_blocks = nn.Sequential()
        self.view_blocks_codes = []
        for view_block in view_blocks:
            self.view_blocks.add_module(str(view_block.code), view_block)
            self.view_blocks_codes.append(str(view_block.code))
        # self.view_blocks = view_blocks
        self.model_args = model_args
        self.comm_feature_num = comm_feature_num
        view_count = len(self.view_blocks)
        self.final_feature_num = (view_count + 1) * comm_feature_num
        self.fc_comm_extract = nn.Linear(comm_feature_num, comm_feature_num)
        self.fc_predictor = nn.Linear(self.final_feature_num, label_num)

        if self.model_args['has_comm_ML_Loss']:
            self.fc_comm_predictor = nn.Linear(self.comm_feature_num, label_num)

        if self.model_args['has_GAN']:
            self.discriminator = nn.Linear(comm_feature_num, view_count)

        if self.model_args['has_contrastive_loss']:
            self.projection_head = nn.Linear(comm_feature_num, projection_dim) # TODO


    def forward(self, x, is_training=True, labels = None):
        view_features_dict = self._extract_view_features(x)
        final_features = torch.zeros(x[0].shape[0], self.final_feature_num)
        comm_feature = 0.0  # common representation vector
        comm_ML_loss = 0.0
        GAN_loss = 0.0

        # add contrastive loss
        contrastive_loss = 0.0

        view_count = len(self.view_blocks)

        comm_predictions = 0.0

        for view_code, view_feature in view_features_dict.items():
            view_code = int(view_code)
            final_features[:, view_code * self.comm_feature_num: (view_code + 1) *
            self.comm_feature_num] = view_feature[0]
            view_comm_feature = self.fc_comm_extract(view_feature[1])

            if self.model_args['has_GAN'] and is_training:
                GAN_loss += self._calculate_GAN_loss(view_comm_feature, view_code)

            if self.model_args['has_comm_ML_Loss']:
                comm_prediction = self._calculate_Comm_ML_predicitons(view_comm_feature)
                if is_training:
                    comm_loss = self._calculate_Comm_ML_loss(comm_prediction, labels)
                    comm_ML_loss += comm_loss
                else:
                    comm_predictions += torch.sigmoid(comm_prediction)

            # If contrastive loss is enabled, project features
            if self.model_args['has_contrastive_loss']:
                # print("calculating contrastive_loss")
                # TODO
                projected_feature = self.projection_head(view_comm_feature)
                positive_pairs, negative_pairs = self.generate_contrastive_pairs(projected_feature, num_negatives=5)

                for pos_pair in positive_pairs:
                    anchor = pos_pair[0].unsqueeze(0)
                    positive = pos_pair[1].unsqueeze(0)
                    pos_loss = self.contrastive_loss_function(anchor, positive)
                    contrastive_loss += pos_loss

                for neg_pair in negative_pairs:
                    anchor = neg_pair[0].unsqueeze(0)
                    negative = neg_pair[1].unsqueeze(0)
                    neg_loss = self.contrastive_loss_function(positive, negative)
                    contrastive_loss += neg_loss

            comm_feature += view_comm_feature

        comm_feature /= view_count

        final_features[:, -self.comm_feature_num:] = comm_feature

        label_predictions = self.fc_predictor(final_features)

        if is_training:
            train_return = {}
            train_return['label_predictions'] = label_predictions
            if self.model_args['has_orthogonal_regularization']:
                if self.model_args['regularization_type'] == 'L1':
                    orthogonal_regularization = self._calculate_orthogonal_regularization_L1(
                        view_features_dict, comm_feature)
                elif self.model_args['regularization_type'] == 'L2':
                    orthogonal_regularization = self._calculate_orthogonal_regularization_L2(
                        view_features_dict, comm_feature)
                else:
                    orthogonal_regularization = self._calculate_orthogonal_regularization_F(
                        view_features_dict, comm_feature)
                orthogonal_regularization /= view_count
                train_return['orthogonal_regularization'] = orthogonal_regularization

            if self.model_args['has_GAN']:
                GAN_loss /= view_count
                GAN_loss = torch.exp(-GAN_loss)
                train_return['GAN_loss'] = GAN_loss

            if self.model_args['has_comm_ML_Loss']:
                comm_ML_loss /= view_count
                train_return['comm_ML_loss'] = comm_ML_loss

            if self.model_args['has_contrastive_loss']:
                # print("add contrastive_loss")
                # TODO
                train_return['contrastive_loss'] = contrastive_loss / view_count

            return train_return

        inference_return = {}
        inference_return['label_predictions'] = label_predictions
        if self.model_args['has_comm_ML_Loss']:
            comm_predictions /= view_count
            inference_return['comm_label_predictions'] = comm_predictions

        return inference_return

    def generate_contrastive_pairs(self, projected_features, num_negatives=3):
        num_samples = projected_features.size(0)
        positive_pairs = []
        negative_pairs = []
        num_negatives = min(num_negatives, num_samples - 1)
        for i in range(num_samples):
            # views = projected_features[i]

            # 正样本对：同一个样本的不同视图
            positive_pair = (projected_features[i], projected_features[i])
            positive_pairs.append(positive_pair)

            # 负样本对：随机选择其他样本作为负样本
            negative_indices = random.sample(range(num_samples), num_negatives)
            for neg_index in negative_indices:
                if neg_index != i:
                    negative_pair = (projected_features[i], projected_features[neg_index])
                    negative_pairs.append(negative_pair)

        return positive_pairs, negative_pairs


    def _extract_view_features(self, x):
        view_features_dict = {}
        for view_blcok_code in self.view_blocks_codes:
            view_x = x[int(view_blcok_code)]
            view_block = self.view_blocks.__getattr__(view_blcok_code)
            view_features = view_block(view_x)
            view_features_dict[view_blcok_code] = view_features 
        return view_features_dict

    def _calculate_Comm_ML_predicitons(self, view_comm_feature):
        return self.fc_comm_predictor(view_comm_feature)

    def _calculate_Comm_ML_loss(self,label_predictions, labels):
        ML_loss = My_logit_ML_loss(label_predictions, labels)
        return ML_loss

    def _calculate_GAN_loss(self, view_comm_feature, code):
        pre_distributions = self.discriminator(view_comm_feature)
        true_distributions = torch.zeros(pre_distributions.shape)
        true_distributions[:, code] = 1.0
        loss = My_KL_loss(pre_distributions, true_distributions)
        return loss

    def _calculate_orthogonal_regularization_L1(self, view_features_dict, comm_feature):
        loss = 0.0
        for _, view_feature in view_features_dict.items():
            item = view_feature[0] * comm_feature
            item = item.sum(1)
            item = torch.abs(item)
            item = item.sum()
            loss += item
        loss /= comm_feature.shape[0]
        return loss

    def _calculate_orthogonal_regularization_L2(self, view_features_dict, comm_feature):
        loss = 0.0
        for _, view_feature in view_features_dict.items():
            item = view_feature[0] * comm_feature
            item = item.sum(1)
            item = item ** 2
            item = item.sum()
            # item = item ** 0.5
            loss += item
        loss /= comm_feature.shape[0]
        return loss

    def _calculate_orthogonal_regularization_F(self, view_features_dict, comm_feature):
        loss = 0.0
        comm_feature_T = comm_feature.t()
        for _, view_feature in view_features_dict.items():
            item = view_feature[0].mm(comm_feature_T)
            item = item ** 2
            item = item.sum()
            # item = item ** 0.5
            loss += item
        loss /= (comm_feature.shape[0] * comm_feature.shape[0])
        return loss

    # def _calculate_nce_loss(self, anchor, positive, negatives):
    #     """
    #     Calculate the NCE loss.
    #     :param anchor: Tensor of anchor samples.
    #     :param positive: Tensor of positive samples.
    #     :param negatives: Tensor of negative samples.
    #     """
    #     # Normalize the embeddings
    #     anchor = F.normalize(anchor, p=2, dim=1)
    #     positive = F.normalize(positive, p=2, dim=1)
    #     negatives = F.normalize(negatives, p=2, dim=1)
    #
    #     # Compute similarity scores
    #     positive_score = torch.sum(anchor * positive, dim=1)
    #     negative_scores = torch.matmul(anchor, negatives.T)
    #
    #     # Calculate the NCE loss
    #     # You can adjust the temperature parameter as needed
    #     temperature = 0.1
    #     logits = torch.cat([positive_score.unsqueeze(1), negative_scores], dim=1) / temperature
    #     labels = torch.zeros(logits.shape[0], dtype=torch.long).to(anchor.device)
    #
    #     loss = F.cross_entropy(logits, labels)
    #     return loss

    def contrastive_loss_function(self, anchor, other, temperature=0.1):
        """
        Calculate contrastive loss for a single pair (anchor and other).
        param anchor (torch.Tensor): Anchor feature vector.
        param other (torch.Tensor): Other feature vector (can be either positive or negative).
        param temperature (float): Temperature parameter for scaling the logits.

        Returns:
        torch.Tensor: Contrastive loss value for the pair.
        """
        # Normalize the features
        anchor = F.normalize(anchor, p=2, dim=1)
        other = F.normalize(other, p=2, dim=1)

        # Compute similarity score
        similarity_score = torch.sum(anchor * other, dim=1) / temperature

        # Creating logits and labels
        logits = torch.cat([similarity_score.unsqueeze(1), torch.zeros_like(similarity_score).unsqueeze(1)], dim=1)
        labels = torch.zeros_like(similarity_score, dtype=torch.long)

        # Calculate the loss
        loss = F.cross_entropy(logits, labels)
        return loss
