import torch.nn as nn
import torch

class BilinearMatrixAttention(nn.Module):
  """
  Adopted from AllenNLP. For now there is no activation function
  """
  def __init__(
        self,
        matrix_1_dim: int,
        matrix_2_dim: int,
        use_input_biases: bool = False,
        label_dim: int = 1) -> None:
    super().__init__()
    if use_input_biases:
      matrix_1_dim += 1
      matrix_2_dim += 1

    if label_dim == 1:
      self.weight_matrix = nn.Parameter(torch.Tensor(matrix_1_dim, matrix_2_dim))
    else:
      self.weight_matrix = nn.Parameter(torch.Tensor(label_dim, matrix_1_dim, matrix_2_dim))
    self.bias = nn.Parameter(torch.Tensor(1))
    self.use_input_biases = use_input_biases

    self.reset_parameters()

  def reset_parameters(self):
    torch.nn.init.xavier_uniform_(self.weight_matrix)
    self.bias.data.fill_(0)

  def forward(self, matrix_1: torch.Tensor, matrix_2: torch.Tensor) -> torch.Tensor:
    if self.use_input_biases:
      bias1 = matrix_1.new_ones(matrix_1.size()[:-1] + (1,))
      bias2 = matrix_2.new_ones(matrix_2.size()[:-1] + (1,))

      matrix_1 = torch.cat([matrix_1, bias1], -1)
      matrix_2 = torch.cat([matrix_2, bias2], -1)

    weight = self.weight_matrix
    if weight.dim() == 2:
      weight = weight.unsqueeze(0)
    intermediate = torch.matmul(matrix_1.unsqueeze(1), weight)
    final = torch.matmul(intermediate, matrix_2.unsqueeze(1).transpose(2, 3))
    return final.squeeze(1) + self.bias



