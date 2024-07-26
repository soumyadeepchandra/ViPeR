import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import os
import pandas as pd

import copy
import numpy as np
import math
import logger as logging

from torchvision import models
logger = logging.get_logger(__name__)

from eval import segment_bars_with_confidence

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def exponential_descrease(idx_decoder, p=3):
    return math.exp(-p*idx_decoder)

class AttentionHelper(nn.Module):
    def __init__(self):
        super(AttentionHelper, self).__init__()
        self.softmax = nn.Softmax(dim=-1)


    def scalar_dot_att(self, proj_query, proj_key, proj_val, padding_mask):
        '''
        scalar dot attention.
        :param proj_query: shape of (B, C, L) => (Batch_Size, Feature_Dimension, Length)
        :param proj_key: shape of (B, C, L)
        :param proj_val: shape of (B, C, L)
        :param padding_mask: shape of (B, C, L)
        :return: attention value of shape (B, C, L)
        '''
        m, c1, l1 = proj_query.shape
        m, c2, l2 = proj_key.shape
        
        assert c1 == c2
        
        energy = torch.bmm(proj_query.permute(0, 2, 1), proj_key)  # out of shape (B, L1, L2)
        attention = energy / np.sqrt(c1)
        attention = attention + torch.log(padding_mask + 1e-6) # mask the zero paddings. log(1e-6) for zero paddings
        attention = self.softmax(attention) 
        attention = attention * padding_mask
        attention = attention.permute(0,2,1)
        out = torch.bmm(proj_val, attention)
        return out, attention

class AttLayer(nn.Module):
    def __init__(self, q_dim, k_dim, v_dim, r1, r2, r3, bl, stage, att_type): # r1 = r2
        super(AttLayer, self).__init__()
        
        self.query_conv = nn.Conv1d(in_channels=q_dim, out_channels=q_dim // r1, kernel_size=1)
        self.key_conv = nn.Conv1d(in_channels=k_dim, out_channels=k_dim // r2, kernel_size=1)
        self.value_conv = nn.Conv1d(in_channels=v_dim, out_channels=v_dim // r3, kernel_size=1)
        
        self.conv_out = nn.Conv1d(in_channels=v_dim // r3, out_channels=v_dim, kernel_size=1)

        self.bl = bl
        self.stage = stage
        self.att_type = att_type
        assert self.att_type in ['normal_att']
        assert self.stage in ['encoder','decoder']
        
        self.att_helper = AttentionHelper()
        self.window_mask = self.construct_window_mask()
        
    
    def construct_window_mask(self):
        '''
            construct window mask of shape (1, l, l + l//2 + l//2), used for sliding window self attention
        '''
        window_mask = torch.zeros((1, self.bl, self.bl + 2* (self.bl //2)))
        for i in range(self.bl):
            window_mask[:, i, i:i+self.bl] = 1
        return window_mask.to(device)
    
    def forward(self, x1, x2, mask):
        # x1 from the encoder
        # x2 from the decoder
        
        query = self.query_conv(x1)
        # key = self.key_conv(x1)
         
        if self.stage == 'decoder':
            assert x2 is not None
            # Here they has Q, K <- encoder and V <- decoder
            # Actual cross attention has Q <- encoder and K, V <- decoder 
            # Adding these modifications
            key = self.key_conv(x2)
            value = self.value_conv(x2)
        else:
            key = self.key_conv(x1)
            value = self.value_conv(x1)
            
        if self.att_type == 'normal_att':
            return self._normal_self_att(query, key, value, mask)

    
    def _normal_self_att(self,q,k,v, mask):
        m_batchsize, c1, L = q.size()
        _,c2,L = k.size()
        _,c3,L = v.size()
        padding_mask = torch.ones((m_batchsize, 1, L)).to(device) * mask[:,0:1,:]
        output, attentions = self.att_helper.scalar_dot_att(q, k, v, padding_mask)
        output = self.conv_out(F.relu(output))
        output = output[:, :, 0:L]
        return output * mask[:, 0:1, :]  
        


class MultiHeadAttLayer(nn.Module):
    def __init__(self, q_dim, k_dim, v_dim, r1, r2, r3, bl, stage, att_type, num_head):
        super(MultiHeadAttLayer, self).__init__()
        #         assert v_dim % num_head == 0
        self.conv_out = nn.Conv1d(v_dim * num_head, v_dim, 1)
        self.layers = nn.ModuleList(
            [copy.deepcopy(AttLayer(q_dim, k_dim, v_dim, r1, r2, r3, bl, stage, att_type)) for i in range(num_head)])
        self.dropout = nn.Dropout(p=0.5)
        
    def forward(self, x1, x2, mask):
        out = torch.cat([layer(x1, x2, mask) for layer in self.layers], dim=1)
        out = self.conv_out(self.dropout(out))
        return out
            

class ConvFeedForward(nn.Module):
    def __init__(self, dilation, in_channels, out_channels):
        super(ConvFeedForward, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 3, padding=dilation, dilation=dilation),
            nn.ReLU()
        )

    def forward(self, x):
        return self.layer(x)


class FCFeedForward(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FCFeedForward, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 1),  # conv1d equals fc
            nn.ReLU(),
            nn.Dropout(),
            nn.Conv1d(out_channels, out_channels, 1)
        )
        
    def forward(self, x):
        return self.layer(x)
    

class AttModule(nn.Module):
    def __init__(self, dilation, in_channels, out_channels, r1, r2, att_type, stage, alpha):
        super(AttModule, self).__init__()
        self.feed_forward = ConvFeedForward(dilation, in_channels, out_channels)
        self.instance_norm = nn.InstanceNorm1d(in_channels, track_running_stats=False)
        self.att_layer = AttLayer(in_channels, in_channels, out_channels, r1, r1, r2, dilation, att_type=att_type, stage=stage) # dilation
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()
        self.alpha = alpha
        
    def forward(self, x, f, mask):
        #print("Input to dilated conv: ", x.shape)
        out = self.feed_forward(x)

        out = self.alpha * self.att_layer(self.instance_norm(out), f, mask) + out

        #print("Input to conv1x1: ", out.shape)
        out = self.conv_1x1(out)
        out = self.dropout(out)
        #print("Output of conv1x1: ", out.shape)
        return (x + out) * mask[:, 0:1, :]


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, max_len=10000):
        super(PositionalEncoding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).permute(0,2,1) # of shape (1, d_model, l)
        self.pe = nn.Parameter(pe, requires_grad=True)


    def forward(self, x):
        return x + self.pe[:, :, 0:x.shape[2]]

class Outputlayer(nn.Module):   # Adding a residual connection where final feature output is a weighted sum of output of each stage of encoder/decoder
    def __init__(self, num_layers):
        super(Outputlayer, self).__init__()
        self.weights = nn.Parameter(torch.rand(num_layers, 1, 64, 1))
        self.bias = nn.Parameter(torch.rand(1, 1, 64, 1))
    
    def forward(self, inputs):
        output = torch.sum(self.weights * inputs, dim=0, keepdim=True) + self.bias
        # print("Input shape: ", inputs.shape)
        # print("Output shape: ", output.shape)
        return torch.relu(output.squeeze(0))

class Encoder(nn.Module):
    def __init__(self, num_layers, r1, r2, num_f_maps, input_dim, num_classes, channel_masking_rate, att_type, alpha):
        super(Encoder, self).__init__()
        self.num_layers = num_layers
        self.conv_1x1 = nn.Conv1d(input_dim, num_f_maps, 1) # fc layer
        self.layers = nn.ModuleList(
            [AttModule(2 ** i, num_f_maps, num_f_maps, r1, r2, att_type, 'encoder', alpha) for i in # 2**i
             range(num_layers)])
        
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)
        self.dropout = nn.Dropout2d(p=channel_masking_rate)
        self.weight_feature = Outputlayer(num_layers)
        self.channel_masking_rate = channel_masking_rate

    def forward(self, x, mask):
        '''
        :param x: (N, C, L)
        :param mask:
        :return:
        '''
        #print("\n Encoder input: ", x.shape)
        if self.channel_masking_rate > 0:
            x = x.unsqueeze(2)
            x = self.dropout(x)
            x = x.squeeze(2)

        feature = self.conv_1x1(x)
        #print("Output after Conv 1x1: ", feature.shape)
        
        feature_w = torch.zeros(self.num_layers, feature.shape[0], feature.shape[1], feature.shape[2], dtype=torch.float).to(device)
        for idx, layer in enumerate(self.layers):
            feature = layer(feature, None, mask)
            feature_w[idx, :, :, : ] = feature
            #print("Output after attention block: ", feature.shape)
        
        out = self.conv_out(feature) * mask[:, 0:1, :]
        #print("Encoder output: ", out.shape)
        feature_res = self.weight_feature(feature_w) 
        # print("feature output: ", feature.shape)
        # print("feature_w output: ", feature_w.shape)
        # print("feature_res output: ", feature_res.shape)
        return out, feature_res


class Decoder(nn.Module):
    def __init__(self, num_layers, r1, r2, num_f_maps, input_dim, num_classes, att_type, alpha):
        super(Decoder, self).__init__()
        self.num_layers = num_layers
        self.conv_1x1 = nn.Conv1d(input_dim, num_f_maps, 1)
        self.layers = nn.ModuleList(
            [AttModule(2 ** i, num_f_maps, num_f_maps, r1, r2, att_type, 'decoder', alpha) for i in # 2 ** i
             range(num_layers)])
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)
        self.weight_feature = Outputlayer(num_layers)

    def forward(self, x, fencoder, mask):
        #print("\n Decoder input: ", x.shape, fencoder.shape)
        feature = self.conv_1x1(x)
        #print("Output after Conv 1x1: ", feature.shape)

        feature_w = torch.zeros(self.num_layers, feature.shape[0], feature.shape[1], feature.shape[2], dtype=torch.float).to(device)
        for idx, layer in enumerate(self.layers):
            feature = layer(feature, fencoder, mask)
            feature_w[idx, :, :, :] = feature
            #print("Output after attention block: ", feature.shape)

        out = self.conv_out(feature) * mask[:, 0:1, :]
        #print("Decoder output: ", out.shape) 
        feature_res = self.weight_feature(feature_w) 
        return out, feature_res

class MyTransformer(nn.Module):
    def __init__(self, num_decoders, num_layers, r1, r2, num_f_maps, input_dim, num_classes, channel_masking_rate):
        super(MyTransformer, self).__init__()
        self.encoder = Encoder(num_layers, r1, r2, num_f_maps, input_dim, num_classes, channel_masking_rate, att_type='normal_att', alpha=1)
        self.decoders = nn.ModuleList([Decoder(num_layers, r1, r2, num_f_maps, num_classes, num_classes, att_type='normal_att', alpha=exponential_descrease(s)) for s in range(num_decoders)]) # num_decoders
        
        
    def forward(self, x, mask):

        out, feature = self.encoder(x, mask)
        # print("Encoder input: ", x.shape)
        # print("Encoder output: ", out.shape)  #[b, task(t), f]
        # print("Encoder feature output: ", feature.shape) #[b, 64, f]

        outputs = out.unsqueeze(0)   # [1, b, t, f]
        # print("Encoder output unsqueezed/Decoder Input: ", outputs.shape)

        for decoder in self.decoders:
            out, feature = decoder(F.softmax(out, dim=1) * mask[:, 0:1, :], feature* mask[:, 0:1, :], mask)
            # print("Decoder output: ", out.shape) #[b, task, f]
            # print("Decoder feature output: ", feature.shape)  #[b, 64, f]
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)   # [en+dec, b, t, f]

        # print("Decoder Output: ", outputs.shape)

        return outputs

    
class Trainer:
    def __init__(self, num_layers, r1, r2, num_f_maps, input_dim, num_classes, channel_masking_rate):
        
        self.model = MyTransformer(3, num_layers, r1, r2, num_f_maps, input_dim, num_classes, channel_masking_rate)
        self.ce = nn.CrossEntropyLoss(ignore_index=-100)
        # parameters1 = sum(p.numel() for p in self.res_model.parameters())/1000000
        parameters = sum(p.numel() for p in self.model.parameters())/1000000
        logger.critical('Model Size: Transformer:{} M, ' .format(parameters))
        self.mse = nn.MSELoss(reduction='none')
        self.num_classes = num_classes

    def train(self, save_dir, batch_gen, num_epochs, batch_size, learning_rate, batch_gen_tst=None):
        
        
        self.model.train()
        self.model.to(device)
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        logger.critical('LR:{}'.format(learning_rate))
        
        # ckpt_name= save_dir + "/epoch-0.8870004270662709.model"
        # ckpt_opt = save_dir + "/epoch-0.8870004270662709.opt"
        # print('##### Loaded model from {}\n'.format(ckpt_name))

        # self.model.load_state_dict(torch.load(ckpt_name))
        # optimizer.load_state_dict(torch.load(ckpt_opt))

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
        for epoch in range(num_epochs):
            epoch_loss = 0
            correct = 0
            total = 0
            best_acc = 0
            best_epoch = 0
            while batch_gen.has_next():
                batch_input, batch_target, mask, vids = batch_gen.next_batch(batch_size)

                # print("batch_input shape: ", batch_input.shape) # [b, f, c]
                # print("batch_target shape: ", batch_target.shape) # [b, f]

                batch_input, batch_target, mask = batch_input.to(device), batch_target.to(device), mask.to(device)
                optimizer.zero_grad()

                batch_input = batch_input.permute(0, 2, 1).to(device)  # convert output [b, f, c] -> [b, c, f] for asformer
                # print("batch_input shape: ", batch_input.shape)  # [b, c, f]
                # print("batch_target shape: ", batch_target.shape)  # [b, f]

                ps = self.model(batch_input, mask)

                # print("ps shape: ", ps.shape)   #[enc+dec, b, task(t), f]

                loss = 0
                for p in ps:
                    # print("p in ps shape: ", p.shape)
                    # print("p.transpose(2, 1).contiguous().view(-1, self.num_classes): ", p.transpose(2, 1).contiguous().view(-1, self.num_classes))
                    # print("batch_target.view(-1): ", batch_target.view(-1))

                    # print("p[:, :, 1:] = ", p[:, :, 1:])

                    loss += self.ce(p.transpose(2, 1).contiguous().view(-1, self.num_classes), batch_target.view(-1))
                    loss += 0.15 * torch.mean(torch.clamp(
                        self.mse(F.log_softmax(p[:, :, 1:], dim=1), F.log_softmax(p.detach()[:, :, :-1], dim=1)), min=0,
                        max=16) * mask[:, :, 1:])

                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()

                # print("ps.data[-1]: ", ps.data[-1])
                # print("Predicted: torch.max(ps.data[-1], 1): ", torch.max(ps.data[-1], 1))

                
                _, predicted = torch.max(ps.data[-1], 1)
                correct += ((predicted == batch_target).float() * mask[:, 0, :].squeeze(1)).sum().item()
                total += torch.sum(mask[:, 0, :]).item()

                temp1 = ((predicted == batch_target).float() * mask[:, 0, :].squeeze(1)).sum().item()
                temp2 = torch.sum(mask[:, 0, :]).item()
                print("Train: {}: Correct: {}/{}".format(vids, temp1, temp2))

                batch_input.detach()
                batch_target.detach()
                mask.detach()
                del batch_input
                del batch_target
                del mask
                # print("total += torch.sum(mask[:, 0, :]).item())", torch.sum(mask[:, 0, :]).item())
            
            scheduler.step(epoch_loss)
            batch_gen.reset()
            # print("[epoch %d]: epoch loss = %f,   acc = %f" % (epoch + 1, epoch_loss / len(batch_gen.list_of_examples),
            #                                                    float(correct) / total))
            
            logger.critical("[epoch {}]: epoch loss = {},   acc = {}" .format(epoch + 1, epoch_loss / len(batch_gen.list_of_examples),
                                                               float(correct) / total))                                                               

            if (epoch + 1) % 10 == 0 and batch_gen_tst is not None:
                acc = self.test(batch_gen_tst, epoch)
                # torch.save(self.model.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + "-" + str(acc) + ".model")
                # torch.save(optimizer.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + "-" + str(acc) + ".opt")
                # logger.critical("Current Test Accuracy: [epoch: {}] is {}".format(str(epoch + 1), str(acc)))
                if(acc > best_acc):
                    best_acc = acc
                    best_epoch = epoch
                    torch.save(self.model.state_dict(), save_dir + "/epoch-" + str(best_epoch + 1)+ "-" + str(best_acc) + ".model")
                    torch.save(optimizer.state_dict(), save_dir + "/epoch-" + str(best_epoch + 1)+ "-" + str(best_acc) + ".opt")
                logger.critical("Best Test Accuracy: [epoch: {}] is {}".format(str(best_epoch + 1), str(best_acc)))

    def test(self, batch_gen_tst, epoch):
        self.model.eval()
        correct = 0
        total = 0
        if_warp = False  # When testing, always false
        with torch.no_grad():
            while batch_gen_tst.has_next():
                batch_input, batch_target, mask, vids = batch_gen_tst.next_batch(1)
                batch_input, batch_target, mask = batch_input.to(device), batch_target.to(device), mask.to(device)
                
                # batch_input = self.res_model(batch_input[0])   # take one input dataloader [f, channel, h , w] -> [f, c]
                batch_input = batch_input.permute(0, 2, 1).to(device)  # convert output [b, f,c] -> [b, c, f] for asformer
                
                p = self.model(batch_input, mask)
                _, predicted = torch.max(p.data[-1], 1)
                correct += ((predicted == batch_target).float() * mask[:, 0, :].squeeze(1)).sum().item()
                total += torch.sum(mask[:, 0, :]).item()

                temp1 = ((predicted == batch_target).float() * mask[:, 0, :].squeeze(1)).sum().item()
                temp2 = torch.sum(mask[:, 0, :]).item()
                logger.critical("Test: {}: Correct: {}/{}".format(vids, temp1, temp2))


        acc = float(correct) / total
        print("---[epoch %d]---: tst acc = %f" % (epoch + 1, acc))

        self.model.train()
        batch_gen_tst.reset()
        return acc

    def predict(self, model_dir, results_dir, features_path, features_path2, anno_path, batch_gen_tst, epoch, actions_dict, sample_rate):
        self.model.eval()
        with torch.no_grad():
            self.model.to(device)
            # self.model.load_state_dict(torch.load("/home/nano01/a/chand133/Urology/ViT_model/models/QK_V_res_run3/epoch-170-0.6035686998599261.model"))
            self.model.load_state_dict(torch.load("./models/epoch-200-0.6197886158156118.model"))
            
            batch_gen_tst.reset()
            import time
            
            time_start = time.time()
            while batch_gen_tst.has_next():
                batch_input, batch_target, mask, vids = batch_gen_tst.next_batch(1)
                print("batch_input shape: ", batch_input.shape) # [b, f, c, h, w]
                print("batch_target shape: ", batch_target.shape) # [b, f]
                vid = vids[0]
                print(vid)

                # FEATURES (B, F, 2048)
                features = np.load(features_path + vid.split('.')[0] + '.npy')
                
                # print("Shape of features: ", np.shape(features))

                sample_rate = 1
                if np.shape(features)[0] > 50000:
                    sample_rate = 2 
                if np.shape(features)[0] > 100000: 
                    sample_rate = 4 
                if np.shape(features)[0] > 150000: 
                    sample_rate = 6 
                if np.shape(features)[0] > 200000: 
                    sample_rate = 8 
                if np.shape(features)[0] > 250000: 
                    sample_rate = 10 

                features = features[::sample_rate, :]
                features = features.transpose(1,0)
                print("Shape of features: ", np.shape(features))
                
                # FEATURES (B, F, C, H, W)
                features2 = np.load(features_path2 + vid.split('.')[0] + '.npy')
                # print("Shape of features: ", np.shape(features))
                
                video_name = vid.split('.')[0]
                csv_file = os.path.join(anno_path, f'{video_name}.csv')
                annotation_data = pd.read_csv(open(csv_file, 'r'), header=None)

            
                all_ranges = []
                for step in annotation_data.values:
                    start = step[0]
                    end = step[1]
                    diff = step[1] - step[0]
                    row = step[2].split(' ')
                    task = ' '.join(row[1:])
                    all_ranges.append((start,end))
                    

                print(all_ranges)
                
                features_temp = np.concatenate([features2[start:end+1] for start, end in all_ranges])

                features2 = features_temp[::sample_rate, :, :, :]
                #features = features.transpose(1,0,2,3,4)
                print("Shape of features: ", np.shape(features))
                
                input_x = torch.tensor(features, dtype=torch.float)
                input_x.unsqueeze_(0)
                input_x = input_x.to(device)
                predictions = self.model(input_x, torch.ones(input_x.size(), device=device))

                for i in range(len(predictions)):
                    confidence, predicted = torch.max(F.softmax(predictions[i], dim=1).data, 1)
                    confidence, predicted = confidence.squeeze(), predicted.squeeze()
 
                    batch_target = batch_target.squeeze()
                    confidence, predicted = confidence.squeeze(), predicted.squeeze()
 
                    segment_bars_with_confidence(results_dir + '/{}_stage{}.png'.format(vid, i),
                                                 confidence.tolist(),
                                                 batch_target.tolist(), predicted.tolist())

                recognition = []
                for i in range(len(predicted)):
                    recognition = np.concatenate((recognition, [list(actions_dict.keys())[
                                                                    list(actions_dict.values()).index(
                                                                        predicted[i].item())]] * sample_rate))
                f_name = vid.split('/')[-1].split('.')[0]
                f_ptr = open(results_dir + "/" + f_name, "w")
                f_ptr.write("### Frame level recognition: ###\n")
                f_ptr.write(' '.join(recognition))
                f_ptr.close()
            time_end = time.time()
            
            

if __name__ == '__main__':
    pass
