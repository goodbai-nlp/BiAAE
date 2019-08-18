#!/usr/bin/env python
# encoding: utf-8


"""
@version: ??
@author: muyeby
@contact: bxf_hit@163.com
@site: http://muyeby.github.io
@software: PyCharm
@file: trainer.py.py
@time: 2018/10/11 9:53
"""

import sys
import torch
import numpy as np
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
from model import AE,Discriminator
from timeit import default_timer as timer
import matplotlib
from torch.autograd import Variable
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import io
import random
from datetime import timedelta
import json
import copy
from word_translation import get_word_translation_accuracy
from dico_builder import build_dictionary,dist_mean_cosine

class BiAAE(object):
    def __init__(self, params):

        self.params = params
        self.tune_dir = "{}/{}-{}/{}".format(params.exp_id,params.src_lang,params.tgt_lang,params.norm_embeddings)
        self.tune_best_dir = "{}/best".format(self.tune_dir)

        self.X_AE = AE(params)
        self.Y_AE = AE(params)
        self.D_X = Discriminator(input_size=params.d_input_size, hidden_size=params.d_hidden_size,
                            output_size=params.d_output_size)
        self.D_Y = Discriminator(input_size=params.d_input_size, hidden_size=params.d_hidden_size,
                            output_size=params.d_output_size)

        self.nets = [self.X_AE, self.Y_AE, self.D_X, self.D_Y]
        self.loss_fn = torch.nn.BCELoss()
        self.loss_fn2 = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

    def weights_init(self, m):  # 正交初始化
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.orthogonal(m.weight)
            if m.bias is not None:
                torch.nn.init.constant(m.bias, 0.01)

    def weights_init2(self, m):  # xavier_normal 初始化
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_normal(m.weight)
            if m.bias is not None:
                torch.nn.init.constant(m.bias, 0.01)

    def weights_init3(self, m):  # 单位阵初始化
        if isinstance(m, torch.nn.Linear):
            m.weight.data.copy_(torch.diag(torch.ones(self.params.g_input_size)))

    def freeze(self, m):
        for p in m.parameters():
            p.requires_grad = False

    def defreeze(self, m):
        for p in m.parameters():
            p.requires_grad = True


    def init_state(self,seed=-1):
        if torch.cuda.is_available():
            # Move the network and the optimizer to the GPU
            for net in self.nets:
                net.cuda()
            self.loss_fn = self.loss_fn.cuda()
            self.loss_fn2 = self.loss_fn2.cuda()

        print('Init3 the model...')
        self.X_AE.apply(self.weights_init)  # 可更改G初始化方式
        self.Y_AE.apply(self.weights_init)  # 可更改G初始化方式
        
        self.D_X.apply(self.weights_init2)
        #print(self.D_X.map1.weight)
        self.D_Y.apply(self.weights_init2)

    def train(self, src_dico, tgt_dico, src_emb, tgt_emb, seed):
        # Load data
        if not os.path.exists(self.params.data_dir):
            print("Data path doesn't exists: %s" % self.params.data_dir)
        if not os.path.exists(self.tune_dir):
            os.makedirs(self.tune_dir)
        if not os.path.exists(self.tune_best_dir):
            os.makedirs(self.tune_best_dir)
        
        src_word2id = src_dico[1]
        tgt_word2id = tgt_dico[1]
        en = src_emb
        it = tgt_emb

        #eval = Evaluator(self.params, en,it, torch.cuda.is_available())


        AE_optimizer = optim.SGD(filter(lambda p: p.requires_grad, list(self.X_AE.parameters()) + list(self.Y_AE.parameters())), lr=self.params.g_learning_rate)
        D_optimizer = optim.SGD(list(self.D_X.parameters()) + list(self.D_Y.parameters()), lr=self.params.d_learning_rate)

        D_A_acc_epochs = []
        D_B_acc_epochs = []
        D_A_loss_epochs = []
        D_B_loss_epochs = []
        d_loss_epochs = []
        G_AB_loss_epochs = []
        G_BA_loss_epochs = []
        G_AB_recon_epochs = []
        G_BA_recon_epochs = []
        g_loss_epochs = []
        L_Z_loss_epoches = []

        acc_epochs = []

        criterion_epochs = []
        best_valid_metric = -100


        try:
            for epoch in range(self.params.num_epochs):
                D_A_losses = []
                D_B_losses = []
                G_AB_losses = []
                G_AB_recon = []
                G_BA_losses = []
                G_adv_losses = []
                G_BA_recon = []
                L_Z_losses = []
                d_losses = []
                g_losses = []
                hit_A = 0
                hit_B = 0
                total = 0
                start_time = timer()
                # lowest_loss = 1e5
                label_D = to_variable(torch.FloatTensor(2 * self.params.mini_batch_size).zero_())
                label_D[:self.params.mini_batch_size] = 1 - self.params.smoothing
                label_D[self.params.mini_batch_size:] = self.params.smoothing

                label_G = to_variable(torch.FloatTensor(self.params.mini_batch_size).zero_())
                label_G = label_G + 1 - self.params.smoothing

                for mini_batch in range(0, self.params.iters_in_epoch // self.params.mini_batch_size):
                    for d_index in range(self.params.d_steps):
                        D_optimizer.zero_grad()  # Reset the gradients
                        self.D_X.train()
                        self.D_Y.train()

                        view_X, view_Y = self.get_batch_data_fast(en, it)

                        # Discriminator X
                        Y_Z = self.Y_AE.encode(view_Y).detach()
                        fake_X = self.X_AE.decode(Y_Z).detach()
                        input = torch.cat([view_X, fake_X], 0)

                        pred_A = self.D_X(input)
                        D_A_loss = self.loss_fn(pred_A, label_D)

                        # Discriminator Y
                        X_Z = self.X_AE.encode(view_X).detach()
                        fake_Y = self.Y_AE.decode(X_Z).detach()

                        input = torch.cat([view_Y, fake_Y], 0)
                        pred_B = self.D_Y(input)
                        D_B_loss = self.loss_fn(pred_B, label_D)

                        D_loss =  D_A_loss + self.params.gate * D_B_loss

                        D_loss.backward()  # compute/store gradients, but don't change params
                        d_losses.append(to_numpy(D_loss.data))
                        D_A_losses.append(to_numpy(D_A_loss.data))
                        D_B_losses.append(to_numpy(D_B_loss.data))

                        discriminator_decision_A = to_numpy(pred_A.data)
                        hit_A += np.sum(discriminator_decision_A[:self.params.mini_batch_size] >= 0.5)
                        hit_A += np.sum(discriminator_decision_A[self.params.mini_batch_size:] < 0.5)

                        discriminator_decision_B = to_numpy(pred_B.data)
                        hit_B += np.sum(discriminator_decision_B[:self.params.mini_batch_size] >= 0.5)
                        hit_B += np.sum(discriminator_decision_B[self.params.mini_batch_size:] < 0.5)

                        D_optimizer.step()  # Only optimizes D's parameters; changes based on stored gradients from backward()

                        # Clip weights
                        #_clip(self.D_X, self.params.clip_value)
                        #_clip(self.D_Y, self.params.clip_value)

                        sys.stdout.write("[%d/%d] :: Discriminator Loss: %.3f \r" % (
                            mini_batch, self.params.iters_in_epoch // self.params.mini_batch_size,
                            np.asscalar(np.mean(d_losses))))
                        sys.stdout.flush()

                    total += 2 * self.params.mini_batch_size * self.params.d_steps

                    for g_index in range(self.params.g_steps):
                        # 2. Train G on D's response (but DO NOT train D on these labels)
                        AE_optimizer.zero_grad()
                        self.D_X.eval()
                        self.D_Y.eval()
                        view_X, view_Y = self.get_batch_data_fast(en, it)

                        # Generator X_AE
                        ## adversarial loss
                        X_Z = self.X_AE.encode(view_X)
                        X_recon = self.X_AE.decode(X_Z)
                        Y_fake = self.Y_AE.decode(X_Z)
                        pred_Y = self.D_Y(Y_fake)
                        L_adv_X = self.loss_fn(pred_Y, label_G)

                        L_recon_X = 1.0 - torch.mean(self.loss_fn2(view_X, X_recon))


                        # Generator Y_AE
                        # adversarial loss
                        Y_Z = self.Y_AE.encode(view_Y)
                        Y_recon = self.Y_AE.decode(Y_Z)
                        X_fake = self.X_AE.decode(Y_Z)
                        pred_X = self.D_X(X_fake)
                        L_adv_Y = self.loss_fn(pred_X, label_G)

                        ### autoAE Loss
                        L_recon_Y = 1.0 - torch.mean(self.loss_fn2(view_Y, Y_recon))

                        # cross-lingual Loss
                        L_Z = 1.0 - torch.mean(self.loss_fn2(X_Z, Y_Z))

                        G_loss = self.params.adv_weight * (self.params.gate*L_adv_X + L_adv_Y) + \
                                self.params.mono_weight * (L_recon_X+L_recon_Y) + \
                                self.params.cross_weight * L_Z

                        G_loss.backward()

                        g_losses.append(to_numpy(G_loss.data))
                        G_AB_losses.append(to_numpy(L_adv_X.data))
                        G_BA_losses.append(to_numpy(L_adv_Y.data))
                        G_adv_losses.append(to_numpy(L_adv_Y.data+L_adv_X.data))
                        G_AB_recon.append(to_numpy(L_recon_X.data))
                        G_BA_recon.append(to_numpy(L_recon_Y.data))
                        L_Z_losses.append(to_numpy(L_Z.data))


                        AE_optimizer.step()  # Only optimizes G's parameters

                        sys.stdout.write(
                            "[%d/%d] ::                                     Generator Loss: %.3f \r" % (
                                mini_batch, self.params.iters_in_epoch // self.params.mini_batch_size,
                                np.asscalar(np.mean(g_losses))))
                        sys.stdout.flush()

                '''for each epoch'''

                D_A_acc_epochs.append(hit_A / total)
                D_B_acc_epochs.append(hit_B / total)
                G_AB_loss_epochs.append(np.asscalar(np.mean(G_AB_losses)))
                G_BA_loss_epochs.append(np.asscalar(np.mean(G_BA_losses)))
                D_A_loss_epochs.append(np.asscalar(np.mean(D_A_losses)))
                D_B_loss_epochs.append(np.asscalar(np.mean(D_B_losses)))
                G_AB_recon_epochs.append(np.asscalar(np.mean(G_AB_recon)))
                G_BA_recon_epochs.append(np.asscalar(np.mean(G_BA_recon)))
                L_Z_loss_epoches.append(np.asscalar(np.mean(L_Z_losses)))
                d_loss_epochs.append(np.asscalar(np.mean(d_losses)))
                g_loss_epochs.append(np.asscalar(np.mean(g_losses)))

                print(
                    "Epoch {} : Discriminator Loss: {:.3f}, Discriminator Accuracy: {:.3f}, Generator Loss: {:.3f}, Time elapsed {:.2f} mins".
                        format(epoch, np.asscalar(np.mean(d_losses)), 0.5 * (hit_A + hit_B) / total,
                               np.asscalar(np.mean(g_losses)),
                               (timer() - start_time) / 60))


                if (epoch + 1) % self.params.print_every == 0:
                    # No need for discriminator weights

                    X_Z = self.X_AE.encode(Variable(en)).data
                    Y_Z = self.Y_AE.encode(Variable(it)).data

                    mstart_time = timer()
                    for method in [self.params.eval_method]:
                        results = get_word_translation_accuracy(
                            self.params.src_lang, src_word2id, X_Z,
                            self.params.tgt_lang, tgt_word2id, Y_Z,
                            method=method,
                            dico_eval=self.params.eval_file
                        )
                        acc1 = results[0][1]

                    print('{} takes {:.2f}s'.format(method, timer() - mstart_time))
                    print('Method:{} score:{:.4f}'.format(method, acc1))

                    csls,size = dist_mean_cosine(self.params, X_Z, Y_Z)
                    criterion = size
                    if criterion > best_valid_metric:
                        print("New criterion value: {}".format(criterion))
                        best_valid_metric = criterion
                        fp = open(self.tune_best_dir + "/seed_{}_dico_{}_gate_{}_epoch_{}_acc_{:.3f}.tmp".format(seed,self.params.dico_build,self.params.gate,epoch, acc1), 'w')
                        fp.close()
                        torch.save(self.X_AE.state_dict(),self.tune_best_dir+'/seed_{}_dico_{}_gate_{}_best_X.t7'.format(seed,self.params.dico_build,self.params.gate))
                        torch.save(self.Y_AE.state_dict(),self.tune_best_dir+'/seed_{}_dico_{}_gate_{}_best_Y.t7'.format(seed,self.params.dico_build,self.params.gate))
                        torch.save(self.D_X.state_dict(),self.tune_best_dir+'/seed_{}_dico_{}_gate_{}_best_Dx.t7'.format(seed,self.params.dico_build,self.params.gate))
                        torch.save(self.D_Y.state_dict(),self.tune_best_dir+'/seed_{}_dico_{}_gate_{}__best_Dy.t7'.format(seed,self.params.dico_build,self.params.gate))


                    # Saving generator weights
                    fp = open(self.tune_dir + "/seed_{}_gate_{}_epoch_{}_acc_{:.3f}.tmp".format(seed,self.params.gate,epoch,acc1), 'w')
                    fp.close()

                    acc_epochs.append(acc1)
                    criterion_epochs.append(criterion)

            criterion_fb, epoch_fb = max([(score, index) for index, score in enumerate(criterion_epochs)])
            fp = open(self.tune_best_dir +
                      "/seed_{}_dico_{}_gate_{}_epoch_{}_Acc_{:.3f}_{:.4f}.cslsfb".format(seed,self.params.gate,
                                                                                           self.params.dico_build,
                                                                                           epoch_fb,acc_epochs[epoch_fb], criterion_fb), 'w')
            fp.close()

            # Save the plot for discriminator accuracy and generator loss
            fig = plt.figure()
            plt.plot(range(0, len(D_A_acc_epochs)), D_A_acc_epochs, color='b', label='D_A')
            plt.plot(range(0, len(D_B_acc_epochs)), D_B_acc_epochs, color='r', label='D_B')
            plt.ylabel('D_accuracy')
            plt.xlabel('epochs')
            plt.legend()
            fig.savefig(self.tune_dir + '/seed_{}_D_acc.png'.format(seed))

            fig = plt.figure()
            plt.plot(range(0, len(D_A_loss_epochs)), D_A_loss_epochs, color='b', label='D_A')
            plt.plot(range(0, len(D_B_loss_epochs)), D_B_loss_epochs, color='r', label='D_B')
            plt.ylabel('D_losses')
            plt.xlabel('epochs')
            plt.legend()
            fig.savefig(self.tune_dir + '/seed_{}_D_loss.png'.format(seed))

            fig = plt.figure()
            plt.plot(range(0, len(G_AB_loss_epochs)), G_AB_loss_epochs, color='b', label='G_AB')
            plt.plot(range(0, len(G_BA_loss_epochs)), G_BA_loss_epochs, color='r', label='G_BA')
            plt.ylabel('G_losses')
            plt.xlabel('epochs')
            plt.legend()
            fig.savefig(self.tune_dir + '/seed_{}_G_loss.png'.format(seed))

            fig = plt.figure()
            plt.plot(range(0, len(G_AB_recon_epochs)), G_AB_recon_epochs, color='b', label='G_AB')
            plt.plot(range(0, len(G_BA_recon_epochs)), G_BA_recon_epochs, color='r', label='G_BA')
            plt.ylabel('G_recon_loss')
            plt.xlabel('epochs')
            plt.legend()
            fig.savefig(self.tune_dir + '/seed_{}_G_Recon.png'.format(seed))

            # fig = plt.figure()
            # plt.plot(range(0, len(L_Z_loss_epoches)), L_Z_loss_epoches, color='b', label='L_Z')
            # plt.ylabel('L_Z_loss')
            # plt.xlabel('epochs')
            # plt.legend()
            # fig.savefig(tune_dir + '/seed_{}_L_Z.png'.format(seed))

            fig = plt.figure()
            plt.plot(range(0, len(acc_epochs)), acc_epochs, color='b', label='trans_acc1')
            plt.ylabel('trans_acc')
            plt.xlabel('epochs')
            plt.legend()
            fig.savefig(self.tune_dir + '/seed_{}_trans_acc.png'.format(seed))
            '''
            fig = plt.figure()
            plt.plot(range(0, len(csls_epochs)), csls_epochs, color='b', label='csls')
            plt.ylabel('csls')
            plt.xlabel('epochs')
            plt.legend()
            fig.savefig(self.tune_dir + '/seed_{}_csls.png'.format(seed))
            '''
            fig = plt.figure()
            plt.plot(range(0, len(g_loss_epochs)), g_loss_epochs, color='b', label='G_loss')
            plt.ylabel('g_loss')
            plt.xlabel('epochs')
            plt.legend()
            fig.savefig(self.tune_dir + '/seed_{}_g_loss.png'.format(seed))

            fig = plt.figure()
            plt.plot(range(0, len(d_loss_epochs)), d_loss_epochs, color='b', label='csls')
            plt.ylabel('D_loss')
            plt.xlabel('epochs')
            plt.legend()
            fig.savefig(self.tune_dir + '/seed_{}_d_loss.png'.format(seed))
            plt.close('all')


        except KeyboardInterrupt:
            print("Interrupted.. saving model !!!")
            torch.save(self.X_AE.state_dict(), self.tune_dir+'/X_AE_model_interrupt.t7')
            torch.save(self.Y_AE.state_dict(), self.tune_dir+'/Y_AE_model_interrupt.t7')
            torch.save(self.D_X.state_dict(), self.tune_dir+'/D_X_model_interrupt.t7')
            torch.save(self.D_Y.state_dict(), self.tune_dir+'/D_y_model_interrupt.t7')
            exit()

        return

    def get_batch_data_fast(self, emb_en, emb_it):

        params = self.params
        random_en_indices = torch.LongTensor(params.mini_batch_size).random_(params.most_frequent_sampling_size)
        random_it_indices = torch.LongTensor(params.mini_batch_size).random_(params.most_frequent_sampling_size)
        en_batch = to_variable(emb_en)[random_en_indices.cuda()]
        it_batch = to_variable(emb_it)[random_it_indices.cuda()]

        return en_batch, it_batch

def _init_xavier(m):
    if type(m) == torch.nn.Linear:
        fan_in = m.weight.size()[1]
        fan_out = m.weight.size()[0]
        std = np.sqrt(6.0 / (fan_in + fan_out))
        m.weight.data.normal_(0, std)

def to_variable(tensor, volatile=False):
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return torch.autograd.Variable(tensor, volatile)

def to_numpy(tensor):
    if tensor.is_cuda:
        return tensor.cpu().numpy()
    else:
        return tensor.numpy()

def _clip(d, clip):
    if clip > 0:
        for x in d.parameters():
            x.data.clamp_(-clip, clip)

