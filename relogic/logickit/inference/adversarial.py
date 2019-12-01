import torch.nn as nn
import torch
from relogic.logickit.modules.adversarial.discriminator import LR
import numpy as np
from relogic.logickit.base.configuration import AdversarialConfigs


class Adversarial(nn.Module):
  def __init__(self, config: AdversarialConfigs):
    super().__init__()
    self.config = config

    if self.config.discriminator_type == "LR":
      self.discriminator = LR(config=config)
    else:
      NotImplementedError()

    print("Using Adversarial {}".format(config.type))

    if config.type == "WGAN":
      pass

    if config.type == "GAN" or config.type == "GR":
      # Let make it adam for now.
      self.optim = torch.optim.Adam(self.discriminator.parameters(),
                                    lr=self.config.discriminator_lr)



    self.real = 1
    self.fake = 0
    if config.soft_label and config.type == "GAN":
      self.real = np.random.uniform(0.7, 1.0)
      self.fake = np.random.uniform(0.0, 0.3)
      print("The soft label is {} and {}".format(self.real, self.fake))

    if config.type == "GR":
      self.criterion = nn.CrossEntropyLoss()
    else:
      self.criterion = nn.BCEWithLogitsLoss()

  def loss(self, input, label):
    output = self.discriminator(features=input)
    assert (output.dim() == 2)

    if self.config.type == "WGAN":
      loss = torch.mean(output)
    else:
      if self.config.type == "GAN":
        label = torch.empty(*output.size()).fill_(label).type_as(output)
      elif self.config.type == "GR":
        label = torch.empty(output.size(0)).fill_(label).type_as(output).long()

      loss = self.criterion(output, label)
    return output, loss

  def update(self, real_in, fake_in, real_id, fake_id):
    self.optim.zero_grad()

    if self.config.type == "GAN":
      real_id, fake_id = self.real, self.fake

    real_output, real_loss = self.loss(real_in, real_id)
    fake_output, fake_loss = self.loss(fake_in, fake_id)

    if self.config.type in ["GR", "GAN"]:
      loss = 0.5 * (real_loss + fake_loss)
    else:
      loss = fake_loss - real_loss
    loss.backward()
    self.optim.step()

    real_acc, fake_acc = 0, 0
    if self.config.type in ["GR", "GAN"]:
      real_acc = self.accuracy(real_output, real_id)
      fake_acc = self.accuracy(fake_output, fake_id)

    return real_loss.item(), fake_loss.item(), real_acc, fake_acc

  def accuracy(self, output, label):
    if self.config.type == "GAN":
      if label > 0.5:
        preds = (torch.sigmoid(output) >= 0.5).long().cpu()
      else:
        preds = (torch.sigmoid(output) < 0.5).long().cpu()
      label = 1

    labels = torch.LongTensor([label])
    labels = labels.expand(*preds.size())
    n_correct = preds.eq(labels).sum().item()
    acc = 1.0 * n_correct / output.size(0)
    return acc

  def gen_loss(self, real_in, fake_in, real_id, fake_id):
    """Functions to calculate loss to update the Generator.
       The basic idea is to minimize the loss that can confuse the Discriminator
       to make a wrong decision."""
    if self.config.type == "GAN":
      _, loss = self.loss(fake_in, self.real)
      loss = self.config.scale * loss
    elif self.config.type == "WGAN":
      # clamp parameters to a cube
      for p in self.discriminator.parameters():
        p.data.clamp_(self.config.clip_lower, self.config.clip_upper)
      _, loss = self.loss(fake_in, self.real)
      loss = -self.config.scale * loss
    else:
      raise NotImplementedError()
    return loss

  def forward(self, *input, **kwargs):
    raise NotImplementedError()


