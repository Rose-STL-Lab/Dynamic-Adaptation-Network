import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

############################## Meta-Nets ################################
class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(Block, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size = kernel_size, padding = (kernel_size-1)//2)
#         self.norm = nn.Instance(out_channels, affine = False)
#         self.relu = nn.LeakyReLU()
    
        
    def forward(self, xx, weight = None, bias = None):
        if weight is None:
            out = F.relu(F.instance_norm(self.conv(xx)))
        else:
            out = F.relu(F.instance_norm(F.conv2d(F.pad(xx, (1,1,1,1)), weight, bias)))
        return out
    
class MetaNets(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_dim, kernel_size):
        super(MetaNets, self).__init__()
        self.block_1 = Block(in_channels, hidden_dim, 3)
        self.block_2 = Block(hidden_dim, hidden_dim, 3)
        self.block_3 = Block(hidden_dim, hidden_dim, 3)
        self.block_4 = Block(hidden_dim, hidden_dim, 3)
        self.block_5 = Block(hidden_dim, hidden_dim, 3)
        self.block_final = Block(hidden_dim, out_channels, 3)
        
        self.b_weight = nn.Sequential(
            nn.Linear(1, 128),
            nn.Linear(128, 1)
        )
        
        self.b_bias = nn.Sequential(
            nn.Linear(1, 128),
            nn.Linear(128, 1)
        )
        
        self.d_weight = nn.Sequential(
            nn.Linear(1, 128),
            nn.Linear(128, 1)
        )
        
        self.d_bias = nn.Sequential(
            nn.Linear(1, 128),
            nn.Linear(128, 1)
        )
        
    def clear_grad(self):
        """Sets gradients of all model parameters to zero."""
        for p in self.parameters():
            if p.grad is not None:
                p.grad.data.zero_()
                
    def weight_reset(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            m.reset_parameters()

        
    def embed_key(self, xx_support, yy_support, xx_train):
#         self.b_weight_1.copy_(torch.randn(1, 256))
#         self.b_weight_2.copy_(torch.randn(256, 1))
        
#         self.b_bias_1.reset_parameters()
#         self.b_bias_2.reset_parameters()
        
#         self.d_weight_1.reset_parameters()
#         self.d_weight_2.reset_parameters()   
    
#         self.d_bias_1.reset_parameters()
#         self.d_bias_2.reset_parameters()
        
        self.clear_grad()
        
        loss = 0 
        y = yy_support.transpose(0,1)[0] #for y in yy_support.transpose(0,1):
        h = self.block_1(xx_support)
        h = self.block_2(h)
        h = self.block_3(h)
        h = self.block_4(h)
        h = self.block_5(h)
        im = self.block_final(h)
        #xx_support = torch.cat([xx_support[:, 2:], im], 1)
        loss += F.mse_loss(im, y) 
        loss.backward()#retain_graph = True)
        
        grads_w = []
        grads_b = []
        b_split = []
        w_split = []
        
        grads_w.append(self.block_1.conv.weight.reshape(-1,1))
        grads_b.append(self.block_1.conv.bias)
        w_split.append(grads_w[-1].shape[0])
        b_split.append(grads_b[-1].shape[0])
        
        grads_w.append(self.block_2.conv.weight.reshape(-1,1))
        grads_b.append(self.block_2.conv.bias)
        w_split.append(grads_w[-1].shape[0])
        b_split.append(grads_b[-1].shape[0])
        
        grads_w.append(self.block_3.conv.weight.reshape(-1,1))
        grads_b.append(self.block_3.conv.bias)
        w_split.append(grads_w[-1].shape[0])
        b_split.append(grads_b[-1].shape[0])
        
        grads_w.append(self.block_4.conv.weight.reshape(-1,1))
        grads_b.append(self.block_4.conv.bias)
        w_split.append(grads_w[-1].shape[0])
        b_split.append(grads_b[-1].shape[0])
        
        grads_w.append(self.block_5.conv.weight.reshape(-1,1))
        grads_b.append(self.block_5.conv.bias)
        w_split.append(grads_w[-1].shape[0])
        b_split.append(grads_b[-1].shape[0])
        
        grads_w.append(self.block_final.conv.weight.reshape(-1,1))
        grads_b.append(self.block_final.conv.bias)
        w_split.append(grads_w[-1].shape[0])
        b_split.append(grads_b[-1].shape[0])
        
        meta_in_weight = torch.cat(grads_w, dim = 0)
        meta_in_bias = torch.cat(grads_b, dim = 0).unsqueeze(-1)
        
        meta_out_weight = self.b_weight(meta_in_weight)#[0].squeeze(0))
        #torch.matmul(torch.matmul(meta_in_weight, self.b_weight_1), self.b_weight_2)#_2(self.b_weight_1
        meta_out_bias = self.b_bias(meta_in_bias)
#         print(meta_out_weight.shape)
#         print(meta_out_bias.shape)
        
        meta_out_weight = torch.split(meta_out_weight, w_split)
        meta_out_bias = torch.split(meta_out_bias, b_split)
        
        star_w_1, star_b_1 = meta_out_weight[0].reshape(self.block_1.conv.weight.shape), meta_out_bias[0].squeeze(-1)
        star_w_2, star_b_2 = meta_out_weight[1].reshape(self.block_2.conv.weight.shape), meta_out_bias[1].squeeze(-1)
        star_w_3, star_b_3 = meta_out_weight[2].reshape(self.block_3.conv.weight.shape), meta_out_bias[2].squeeze(-1)
        star_w_4, star_b_4 = meta_out_weight[3].reshape(self.block_4.conv.weight.shape), meta_out_bias[3].squeeze(-1)
        star_w_5, star_b_5 = meta_out_weight[4].reshape(self.block_5.conv.weight.shape), meta_out_bias[4].squeeze(-1)
        star_w_f, star_b_f = meta_out_weight[5].reshape(self.block_final.conv.weight.shape), meta_out_bias[5].squeeze(-1)
        
        self.clear_grad()
        
        keys = []
        for xx in [xx_support, xx_train]:
            h = self.block_1(xx) + self.block_1(xx, star_w_1, star_b_1)
            h = self.block_2(h) + self.block_2(h, star_w_2, star_b_2)
            h = self.block_3(h) + self.block_3(h, star_w_3, star_b_3)
            h = self.block_4(h) + self.block_4(h, star_w_4, star_b_4)
            h = self.block_5(h) + self.block_5(h, star_w_5, star_b_5)
            keys.append(h)
          
#         Ws = [star_w_1, star_b_1, star_w_2, star_b_2, star_w_3, star_b_3, 
#               star_w_4, star_b_4, star_w_5, star_b_5, star_w_f, star_b_f]
        Ws = [star_w_1, star_b_1, star_w_2, star_b_2, star_w_3, star_b_3, star_w_f, star_b_f]

        return keys, Ws
    
    def forward(self, xx_support, yy_support, xx_train, test = False):
        
        self.clear_grad()
        
        keys, Ws = self.embed_key(xx_support, yy_support, xx_train)
        key_mems, x_keys = keys
        
        grad_mems = []
        grad_mems1 = []
        
        grad_memo_weight = []
        grad_memo_bias = []
        for i in range(xx_support.shape[0]):
            self.clear_grad()
            x = xx_support[i:i+1]
            support_loss = 0 
            y = yy_support.transpose(0,1)[0]#for y in yy_support.transpose(0,1):
            h = self.block_1(x)
            h = self.block_2(h)
            h = self.block_3(h)
            h = self.block_4(h)
            h = self.block_5(h)
            im = self.block_final(h)
                #x = torch.cat([x[:, 2:], im], 1)
            support_loss += F.mse_loss(im, y) 
            support_loss.backward()#retain_graph = True)

            grads_w = []
            grads_b = []
            b_split = []
            w_split = []

            grads_w.append(self.block_1.conv.weight.reshape(-1,1))
            grads_b.append(self.block_1.conv.bias)
            w_split.append(grads_w[-1].shape[0])
            b_split.append(grads_b[-1].shape[0])

            grads_w.append(self.block_2.conv.weight.reshape(-1,1))
            grads_b.append(self.block_2.conv.bias)
            w_split.append(grads_w[-1].shape[0])
            b_split.append(grads_b[-1].shape[0])

            grads_w.append(self.block_3.conv.weight.reshape(-1,1))
            grads_b.append(self.block_3.conv.bias)
            w_split.append(grads_w[-1].shape[0])
            b_split.append(grads_b[-1].shape[0])

            grads_w.append(self.block_4.conv.weight.reshape(-1,1))
            grads_b.append(self.block_4.conv.bias)
            w_split.append(grads_w[-1].shape[0])
            b_split.append(grads_b[-1].shape[0])

            grads_w.append(self.block_5.conv.weight.reshape(-1,1))
            grads_b.append(self.block_5.conv.bias)
            w_split.append(grads_w[-1].shape[0])
            b_split.append(grads_b[-1].shape[0])

            grads_w.append(self.block_final.conv.weight.reshape(-1,1))
            grads_b.append(self.block_final.conv.bias)
            w_split.append(grads_w[-1].shape[0])
            b_split.append(grads_b[-1].shape[0])

            meta_in_weight = torch.cat(grads_w, dim = 0)
            meta_in_bias = torch.cat(grads_b, dim = 0).unsqueeze(-1)

            meta_out_weight = self.d_weight(meta_in_weight)
            meta_out_bias = self.d_bias(meta_in_bias)
#     
            
            grad_memo_weight.append(meta_out_weight)
            grad_memo_bias.append(meta_out_bias)

        grad_memo_weight = torch.stack(grad_memo_weight, dim = 0)
        grad_memo_bias = torch.stack(grad_memo_bias, dim = 0)
#         print(key_mems.shape, x_keys.shape)
#         print(grad_memo_weight.shape, grad_memo_bias.shape)
       
#         if test:
#             self.eval()
        preds = []
        for i in range(xx_train.shape[0]):
            xx = xx_train[i:i+1]
            x_key = x_keys[i]
            sc = F.softmax(F.cosine_similarity(key_mems.reshape(key_mems.shape[0], -1), x_key.reshape(1, -1)))
            meta_ws = torch.matmul(sc.unsqueeze(0), grad_memo_weight.squeeze(-1))
            meta_bs= torch.matmul(sc.unsqueeze(0), grad_memo_bias.squeeze(-1))
        
            #print(meta_ws.shape, meta_bs.shape)
            out_weight = torch.split(meta_ws, w_split, dim = 1)
            out_bias = torch.split(meta_bs, b_split, dim = 1)

            star_w_1, star_b_1 = out_weight[0].reshape(self.block_1.conv.weight.shape), out_bias[0].squeeze(-1).squeeze(0)
            star_w_2, star_b_2 = out_weight[1].reshape(self.block_2.conv.weight.shape), out_bias[1].squeeze(-1).squeeze(0)
            star_w_3, star_b_3 = out_weight[2].reshape(self.block_3.conv.weight.shape), out_bias[2].squeeze(-1).squeeze(0)
            star_w_4, star_b_4 = out_weight[3].reshape(self.block_4.conv.weight.shape), out_bias[3].squeeze(-1).squeeze(0)
            star_w_5, star_b_5 = out_weight[4].reshape(self.block_5.conv.weight.shape), out_bias[4].squeeze(-1).squeeze(0)
            star_w_f, star_b_f = out_weight[5].reshape(self.block_final.conv.weight.shape), out_bias[5].squeeze(-1)

            loss = 0 
            pred = []
            for y in yy_support.transpose(0,1):
               # print(xx.shape, star_w_1.shape, star_b_1.shape)
                h = self.block_1(xx) + self.block_1(xx, star_w_1, star_b_1)
                h = self.block_2(h) + self.block_2(h, star_w_2, star_b_2)
                h = self.block_3(h) + self.block_3(h, star_w_3, star_b_3)
                h = self.block_4(h) + self.block_4(h, star_w_4, star_b_4)
                h = self.block_5(h) + self.block_5(h, star_w_5, star_b_5)
                im = self.block_final(h) + self.block_final(h, star_w_f, star_b_f)
                pred.append(im)
                xx = torch.cat([xx[:, im.shape[1]:], im], 1)
            preds.append(torch.cat(pred, dim = 1))
        
        return torch.cat(preds, dim = 0)

#     def train_epoch(self, train_loaders, optimizer, loss_function):
#         train_mse = []
#         k = 0
#         for i, data_loader in enumerate(train_loaders):
#             for t_xx, t_yy, s_xx, s_yy in data_loader:
#                 t_xx = t_xx.to(device)
#                 t_yy = t_yy.to(device)
#                 s_xx = s_xx.to(device)
#                 s_yy = s_yy.to(device)

#                 ims = self.forward(s_xx, s_yy, t_xx)
#                # print(ims.shape, t_yy.shape)
                
#                 loss = loss_function(ims, t_yy.reshape(t_yy.shape[0], -1, t_yy.shape[3], t_yy.shape[4])) 

#                 train_mse.append(loss.item()) 
#                 optimizer.zero_grad()
                
#                 self.cleargrads()
#                 loss.backward()
#                 optimizer.step()
#         train_mse = round(np.sqrt(np.mean(train_mse)),5)
#         return train_mse

#     def eval_epoch(self, valid_loaders, loss_function):
#         self.cleargrads()
#         valid_mse = []
#         preds = []
#         trues = []
#        # with torch.no_grad():
#         for i, data_loader in enumerate(valid_loaders):
#             for t_xx, t_yy, s_xx, s_yy in data_loader:
#                 t_xx = t_xx.to(device)
#                 t_yy = t_yy.to(device)
#                 s_xx = s_xx.to(device)
#                 s_yy = s_yy.to(device)


#                 ims = self.forward(s_xx, s_yy, t_xx)

#                 loss = loss_function(ims, t_yy.reshape(t_yy.shape[0], -1, t_yy.shape[3], t_yy.shape[4])) 

#                 preds.append(ims.cpu().data.numpy().reshape(ims.shape[0],-1,2,ims.shape[2],ims.shape[3]))
#                 trues.append(s_yy.cpu().data.numpy())
#                 valid_mse.append(loss.item())
#         preds = np.concatenate(preds, axis = 0)  
#         trues = np.concatenate(trues, axis = 0)  
#         valid_mse = round(np.sqrt(np.mean(valid_mse)), 5)
#         return valid_mse, preds, trues
