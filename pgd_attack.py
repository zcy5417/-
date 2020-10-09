import copy
import numpy as np
import torch



device='cuda'








##############
#############

# def attack(x_nat,y, model,loss_cls,epsilon=0.3,num_steps=10,step_size=0.01,rand=True):
#
#     model_cp = copy.deepcopy(model)
#     model_cp.eval()
#     for p in model_cp.parameters():
#         p.requires_grad = False
#     x_nat=x_nat.cpu().numpy()
#     if rand:
#         x = x_nat + np.random.uniform(-epsilon, epsilon,
#                                       x_nat.shape).astype('float32')
#     else:
#         x = np.copy(x_nat)
#
#     for i in range(num_steps):
#         x_var=torch.from_numpy(x).to(device)
#         x_var.requires_grad=True
#         # y_var=torch.from_numpy(y).to(device)
#
#         y_pred = model_cp(x_var)
#
#         loss = loss_cls(y_pred, y)
#         loss.backward()
#         grad = x_var.grad.data.cpu().numpy()
#
#         x=x+ step_size * np.sign(grad)
#
#         x = np.clip(x, x_nat - epsilon, x_nat + epsilon)
#         x = np.clip(x, 0, 1)  # ensure valid pixel range
#
#     x=torch.from_numpy(x).to(device)
#     return x

def attack(x_nat,y, model,loss_cls,epsilon=0.03,num_steps=10,step_size=0.01,rand=True):

    model_cp = copy.deepcopy(model)
    model_cp.eval()
    for p in model_cp.parameters():
        p.requires_grad = False
    # x_nat=x_nat.cpu().numpy()
    if rand:
        x = x_nat + torch.Tensor(x_nat.shape).uniform_(-epsilon,epsilon).to(device)
    else:
        x = x_nat

    for i in range(num_steps):
        x_var=x.detach()
        x_var.requires_grad=True
        # y_var=torch.from_numpy(y).to(device)

        y_pred = model_cp(x_var)

        loss = loss_cls(y_pred, y)
        loss.backward()
        grad = x_var.grad.data

        x=x+ step_size * grad.sign()

        x = torch.min(x,  x_nat + epsilon)
        x=torch.max(x_nat - epsilon,x)
        x = torch.clamp(x, 0, 1)  # ensure valid pixel range

    # x=torch.from_numpy(x).to(device)
    return x
