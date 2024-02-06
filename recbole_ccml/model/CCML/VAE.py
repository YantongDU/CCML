import torch.nn as nn
import torch
import torch.nn.functional as F

from recbole.model.layers import MLPLayers


class VAE(nn.Module):

    def __init__(self, config, inputDim, conditionDim):
        super().__init__()
        self.VAEEncoderHiddenSize = config['vae_encoder_hidden_size']
        self.device = config.final_config_dict['device']
        self.inputDim = inputDim
        self.conditionDim = conditionDim
        self.lastHiddenSize = self.VAEEncoderHiddenSize[-1]

        # self.encoder = nn.Sequential(
        #     MLPLayers([self.inputDim + self.conditionDim] + self.VAEEncoderHiddenSize[:-1])
        # )

        self.encoder_task = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU()
        )

        self.encoder_user = nn.Linear(inputDim, 128, bias=False)
        self.encoder_item = nn.Linear(conditionDim, 128, bias=False)


        self.fc_mu = nn.Linear(self.VAEEncoderHiddenSize[-2], self.lastHiddenSize)
        self.fc_log_var = nn.Linear(self.VAEEncoderHiddenSize[-2], self.lastHiddenSize)

        # 翻转MLP维度
        self.VAEEncoderHiddenSize.reverse()
        self.decoder = nn.Sequential(
            MLPLayers([self.lastHiddenSize + self.conditionDim] + self.VAEEncoderHiddenSize[1:] + [1])
        )

    def encode(self, input):
        user_emb = input[0, :self.inputDim]
        user_emb_h = self.encoder_user(user_emb)
        item_emb = input[:, self.inputDim:]
        item_emb_h = self.encoder_item(item_emb)

        dot_product = torch.matmul(item_emb_h, user_emb_h.T)  # 结果是10x1
        weights = torch.softmax(dot_product, dim=0)  # 使用softmax获取权重

        task_representation = torch.sum(weights.unsqueeze(1) * item_emb_h, dim=0, keepdim=True) + user_emb

        hidden_input = self.encoder_task(task_representation)
        #
        # input = input.expand(condition.shape[0], -1)
        # condition_input = torch.cat([input, condition], dim=1)
        # hidden_input = self.encoder(input)

        mu = self.fc_mu(hidden_input)
        log_var = self.fc_log_var(hidden_input)
        return mu, log_var


    def decode(self, z, condition):
        expand_z = z.expand(condition.shape[0], -1)
        condition_z = torch.cat([expand_z, condition], dim=1)
        output = self.decoder(condition_z)
        return output

    def reparameter(self, mu, log_var):
        eps = torch.randn(size=(mu.shape), device=self.device)
        std = torch.sqrt(torch.exp(log_var))
        z = mu + eps * std
        return z

    def forward(self, input):
        condition = input[:, self.inputDim:]
        mu, log_var = self.encode(input)
        z = self.reparameter(mu, log_var)
        output = self.decode(z, condition)
        return output, mu, log_var

    def calculate_loss(self, input, lable):
        output, mu, log_var = self.forward(input)

        vae_reconstruct = F.mse_loss(output, lable.float())
        vae_kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return vae_reconstruct + vae_kl_loss

    def predict(self, input, condition, query_input):
        mu, log_var = self.encode(input, condition)
        z = self.reparameter(mu, log_var)
        output = self.decode(z, condition)
        return output, mu, log_var



