# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from aug_search import AUG_TYPE
import pdb


class Controller(nn.Module):
    def __init__(self, hid_size):
        """
        init
        :param hid_size: hidden units in LSTM.
        """
        super(Controller, self).__init__()
        self.hid_size = hid_size
        self.lstm = torch.nn.LSTMCell(self.hid_size, self.hid_size)

        # Embedding is for mapping action into a predifined dictionary.

        # weight 11
        # op1 len(AUG_TYPE)*11
        # op2 len(AUG_TYPE)*11
        all_possible_inputs = 11 + len(AUG_TYPE)*11
        self.encoder = nn.Embedding(all_possible_inputs, self.hid_size) # action has possibilities of type * magnitude

        # action is consist by both aug type and magnitude.
        self.decoder_type = nn.Linear(self.hid_size, len(AUG_TYPE))
        self.decoder_magnitude = nn.Linear(self.hid_size, 11)     # magnitude is discriticized from 0 to 10.

        self.decoder_weight = nn.Linear(self.hid_size, 11)


    def forward_op(self, x, hidden, batch_size):
        if x is None:
            x = self.initHidden(batch_size)
            embed = x
        else:
            embed = self.encoder(x)
        hx, cx = self.lstm(embed, hidden)

        # decode
        type_logit = self.decoder_type(hx)
        magnitude_logit = self.decoder_magnitude(hx)

        return type_logit, magnitude_logit, (hx, cx)


    def forward_weight(self, x, hidden, batch_size):
        if x is None:
            x = self.initHidden(batch_size)
            embed = x
        else:
            embed = self.encoder(x)
        hx, cx = self.lstm(embed, hidden)

        # decode
        weight_logit = self.decoder_weight(hx)

        return weight_logit, (hx, cx)


    def initHidden(self, batch_size):
        device = next(self.parameters()).device
        return torch.zeros(batch_size, self.hid_size, requires_grad=False).to(device)



    def sample(self, batch_size, sub_policy_num=5, sub_policy_operation=2):

        actions = []

        weights = []
        type_entropies = []
        magnitude_entropies = []
        weight_entropies = []
        selected_type_log_probs = []
        selected_mag_log_probs = []
        selected_weight_log_probs = []

        x = None
        hidden = (self.initHidden(batch_size), self.initHidden(batch_size))


        # extract short-cut weight
        short_cut_logit, hidden = self.forward_weight(x, hidden, batch_size)
        short_cut_prob = F.softmax(short_cut_logit, dim=-1)
        short_cut_log_prob = F.log_softmax(short_cut_logit, dim=-1)
        short_cut_entropy = -(short_cut_log_prob * short_cut_prob).sum(1, keepdim=True)
        short_cut_weight = short_cut_prob.multinomial(1) # bacth_size * 1
        x = short_cut_weight
        x = x.squeeze(1)
        x = x.requires_grad_(False)
        selected_short_cut_log_prob = short_cut_log_prob.gather(1, short_cut_weight.data)

        weights.append(short_cut_weight)
        weight_entropies.append(short_cut_entropy)
        selected_weight_log_probs.append(selected_short_cut_log_prob)



        for i in range(sub_policy_num):
            sub_actions = []
            sub_type_entropies = []
            sub_magnitude_entropies =[]
            sub_selected_type_log_probs = []
            sub_selected_mag_log_probs = []

            operations = []
            for j in range(sub_policy_operation):
                type_logit, magnitude_logit, hidden = self.forward_op(x, hidden, batch_size)
                action_type_prob = F.softmax(type_logit, dim=-1)
                action_magnitude_prob = F.softmax(magnitude_logit, dim=-1)

                # Entropies as regulizer
                action_type_log_prob = F.log_softmax(type_logit, dim=-1)
                action_magnitude_log_prob = F.log_softmax(magnitude_logit, dim=-1)
                sub_type_entropies.append(-(action_type_log_prob * action_type_prob).sum(1, keepdim=True))
                sub_magnitude_entropies.append(-(action_magnitude_log_prob * action_magnitude_prob).sum(1, keepdim=True))

                # Get actions
                action_type = action_type_prob.multinomial(1)   # batch_size * 1
                action_magnitude = action_magnitude_prob.multinomial(1)   # batch_size * 1
                action = torch.cat([action_type, action_magnitude], dim=-1) # batch_size * 2

                sub_actions.append(action)


                x = 11 + action_type*11 + action_magnitude
                operations.append(x)

                x = x.squeeze(1)
                x = x.requires_grad_(False)

                # Get selected log prob, this will used for policy gradient calculation
                selected_type_log_prob = action_type_log_prob.gather(1, action_type.data)  # batch_size * 1
                sub_selected_type_log_probs.append(selected_type_log_prob)

                selected_mag_log_prob = action_magnitude_log_prob.gather(1, action_magnitude.data)
                sub_selected_mag_log_probs.append(selected_mag_log_prob)



            # extract weight
            weight_logit, hidden = self.forward_weight(x, hidden, batch_size)
            weight_prob = F.softmax(weight_logit, dim=-1)
            weight_log_prob = F.log_softmax(weight_logit, dim=-1)
            weight_entropy = -(weight_log_prob * weight_prob).sum(1, keepdim=True)
            weight = weight_prob.multinomial(1) # bacth_size * 1
            x = weight
            x = x.squeeze(1)
            x = x.requires_grad_(False)
            selected_weight_log_prob = weight_log_prob.gather(1, weight.data)
            weights.append(weight)
            weight_entropies.append(weight_entropy)
            selected_weight_log_probs.append(selected_weight_log_prob)




            # Process all these appended sub lists.
            # [2, batch_size, 2] -> [batch_size, 2, 2]
            sub_actions = torch.stack(sub_actions).permute(1,0,2)

            # [2, batch_size, 1] -> [batch_size, 1]
            sub_type_entropies = torch.cat(sub_type_entropies, dim=-1).sum(-1, keepdim=True)
            sub_magnitude_entropies = torch.cat(sub_magnitude_entropies, dim=-1).sum(-1, keepdim=True)

            # [2, batch_size, 1] -> [batch_size, 2]
            sub_selected_type_log_probs = torch.cat(sub_selected_type_log_probs, dim=-1)
            sub_selected_mag_log_probs = torch.cat(sub_selected_mag_log_probs, dim=-1)

            actions.append(sub_actions)
            type_entropies.append(sub_type_entropies)
            magnitude_entropies.append(sub_type_entropies)
            selected_type_log_probs.append(sub_selected_type_log_probs)
            selected_mag_log_probs.append(sub_selected_mag_log_probs)

        # Process all lists
        # [5, batch_size, 2, 2] -> [batch_size, 5, 2, 2]
        actions = torch.stack(actions).permute(1,0,2,3)
        # [5, batch_size, 1] -> [batch_size, 1]
        type_entropies = torch.cat(type_entropies, dim=-1).sum(-1, keepdim=True)
        magnitude_entropies = torch.cat(magnitude_entropies, dim=-1).sum(-1, keepdim=True)
        # [5, batch_size, 2] -> [batch_size, 5, 2]
        selected_type_log_probs = torch.stack(selected_type_log_probs).permute(1,0,2)
        selected_mag_log_probs = torch.stack(selected_mag_log_probs).permute(1,0,2)

        # [6, batch_size, 1] -> [batch_size, 6]
        selected_weight_log_probs = torch.stack(selected_weight_log_probs).squeeze(2).permute(1,0)
        weights = torch.stack(weights).squeeze(2).permute(1,0)
        # [6, batch_size, 1] -> [batch_size, 1]
        weight_entropies = torch.stack(weight_entropies).squeeze(2).permute(1,0).sum(-1, keepdim=True)

        # out:
        # actions         [bacth_size, 5, 2, 2] (type, mag)
        # weights         [batch_size, 6]
        # type_log_prob   [batch_size, 5, 2]
        # mag_log_prob    [batch_size, 5, 2]
        # weight_log_prob [batch_size, 6]
        # type_entropies  [batch_size, 1]
        # mag_entropies   [batch_size, 1]
        # weight_entropies[batch_size, 1]


        return {'op':actions, 'weight':weights}, \
               {'type':selected_type_log_probs, 'magnitude':selected_mag_log_probs, 'weight':selected_weight_log_probs}, \
               {'type':type_entropies, 'magnitude':magnitude_entropies, 'weight':weight_entropies}
