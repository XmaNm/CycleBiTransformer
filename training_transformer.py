import os
import numpy as np
from tqdm import tqdm
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import utils as vutils
from transformer import VQTransformer
from utils import load_data, plot_images
from utils import Generator,Discriminator
from torch.autograd import Variable

netG_A2B = Generator(3, 3)
netG_B2A = Generator(3, 3)
netD_A = Discriminator(3)
netD_B = Discriminator(3)

Tensor = torch.cuda.FloatTensor
target_real = Variable(Tensor(1).fill_(1.0), requires_grad=False)
target_fake = Variable(Tensor(1).fill_(0.0), requires_grad=False)

criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()
class TrainTransformer:
    def __init__(self, args):
        self.model = VQTransformer(args).to(device=args.device)
        self.optim = self.configure_optimizers()

        self.train(args)

    def configure_optimizers(self):
        decay, no_decay = set(), set()
        whitelist_weight_modules = (nn.Linear, )
        blacklist_weight_modules = (nn.LayerNorm, nn.Embedding)

        for mn, m in self.model.transformer.named_modules():
            for pn, p in m.named_parameters():
                fpn = f"{mn}.{pn}" if mn else pn

                if pn.endswith("bias"):
                    no_decay.add(fpn)

                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)

                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)

        no_decay.add("pos_emb")

        param_dict = {pn: p for pn, p in self.model.transformer.named_parameters()}

        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": 0.01},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]

        optimizer = torch.optim.AdamW(optim_groups, lr=4.5e-06, betas=(0.9, 0.95))
        return optimizer

    def train(self, args):
        criterion_identity = torch.nn.L1Loss()
        train_dataset = load_data(args)
        for epoch in range(args.epochs):
            with tqdm(range(len(train_dataset))) as pbar:
                for i, imgs in zip(pbar, train_dataset):
                    self.optim.zero_grad()
                    imgs = imgs.to(device=args.device)
                    logits, targets = self.model(imgs) ##logits is A, target is B
                    #Identity loss
                    sampled_logits = netG_A2B(logits)
                    loss_identity_B = criterion_identity(logits, sampled_logits) * 5.0
                    same_targets = netG_B2A(targets)
                    loss_identity_A = criterion_identity(targets, same_targets) * 5.0

                    fake_target = netG_A2B(logits)
                    pred_fake = netD_B(fake_target)
                    loss_GAN_A2B = criterion_GAN(pred_fake, target_real)

                    fake_logits = netG_B2A(targets)
                    pred_fake = netD_A(fake_logits)
                    loss_GAN_B2A = criterion_GAN(pred_fake, target_real)
                    # Cycle loss
                    recovered_logits = netG_B2A(fake_target)
                    loss_cycle_ABA = criterion_cycle(recovered_logits, logits) * 10.0
                    recovered_target = netG_A2B(fake_logits)
                    loss_cycle_BAB = criterion_cycle(recovered_target, targets) * 10.0
                    # Total loss
                    loss = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB
                    ##loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

                    loss.backward()
                    self.optim.step()
                    pbar.set_postfix(Transformer_Loss=np.round(loss.cpu().detach().numpy().item(), 4))
                    pbar.update(0)
            log, sampled_imgs = self.model.log_images(imgs[0][None])
            vutils.save_image(sampled_imgs, os.path.join("results", f"transformer_{epoch}.jpg"), nrow=4)
            plot_images(log)
            torch.save(self.model.state_dict(), os.path.join("checkpoints", f"transformer_{epoch}.pt"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Vector Quantized")
    parser.add_argument('--latent-dim', type=int, default=256, help='Latent dimension n_z.')
    parser.add_argument('--image-size', type=int, default=256, help='Image height and width.)')
    parser.add_argument('--num-codebook-vectors', type=int, default=1024, help='Number of codebook vectors.')
    parser.add_argument('--beta', type=float, default=0.25, help='Commitment loss scalar.')
    parser.add_argument('--image-channels', type=int, default=3, help='Number of channels of images.')
    parser.add_argument('--dataset-path', type=str, default='./data', help='Path to data.')
    parser.add_argument('--checkpoint-path', type=str, default='./checkpoints/last_ckpt.pt', help='Path to checkpoint.')
    parser.add_argument('--device', type=str, default="cuda", help='Which device the training is on')
    parser.add_argument('--batch-size', type=int, default=20, help='Input batch size for training.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
    parser.add_argument('--learning-rate', type=float, default=2.25e-05, help='Learning rate.')
    parser.add_argument('--beta1', type=float, default=0.5, help='Adam beta param.')
    parser.add_argument('--beta2', type=float, default=0.9, help='Adam beta param.')
    parser.add_argument('--disc-start', type=int, default=10000, help='When to start the discriminator.')
    parser.add_argument('--disc-factor', type=float, default=1., help='Weighting factor for the Discriminator.')
    parser.add_argument('--l2-loss-factor', type=float, default=1., help='Weighting factor for reconstruction loss.')
    parser.add_argument('--perceptual-loss-factor', type=float, default=1., help='Weighting factor for perceptual loss.')

    parser.add_argument('--pkeep', type=float, default=0.5, help='Percentage for how much latent codes to keep.')
    parser.add_argument('--sos-token', type=int, default=0, help='Start of Sentence token.')

    args = parser.parse_args()
    args.dataset_path = r"C:\Users\dome\datasets\medical_image"
    args.checkpoint_path = r".\checkpoints\vq_last_ckpt.pt"

    train_transformer = TrainTransformer(args)


