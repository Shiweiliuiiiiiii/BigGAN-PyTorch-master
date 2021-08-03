from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy

import numpy as np
import math

def add_sparse_args(parser):
    parser.add_argument('--sparse', action='store_true', help='Enable sparse mode. Default: True.')
    parser.add_argument('--dy_mode', type=str, default='', help='dynamic change the sparse connectivity of which model')
    parser.add_argument('--sparse_init', type=str, default='ERK', help='sparse initialization')
    parser.add_argument('--growth', type=str, default='gradient', help='Growth mode. Choose from: momentum, random, random_unfired, and gradient.')
    parser.add_argument('--death', type=str, default='magnitude', help='Death mode / pruning mode. Choose from: magnitude, SET, threshold.')
    parser.add_argument('--redistribution', type=str, default='none', help='Redistribution mode. Choose from: momentum, magnitude, nonzeros, or none.')
    parser.add_argument('--death_rate', type=float, default=0.50, help='The pruning rate / death rate.')
    parser.add_argument('--density', type=float, default=0.3, help='The density of the overall sparse network.')
    parser.add_argument('--update_frequency', type=int, default=2000, metavar='N', help='how many iterations to train between parameter exploration')
    parser.add_argument('--decay_schedule', type=str, default='cosine', help='The decay schedule for the pruning rate. Default: cosine. Choose from: cosine, linear.')
    parser.add_argument('--ratio_G', type=float, default=0.50, help='The density ratio of G.')

def get_model_params(model):
    params = {}
    for name in model.state_dict():
        params[name] = copy.deepcopy(model.state_dict()[name])
    return params

def set_model_params(model, model_parameters):
    model.load_state_dict(model_parameters)

class CosineDecay(object):
    def __init__(self, death_rate, T_max, eta_min=0.005, last_epoch=-1):
        self.sgd = optim.SGD(torch.nn.ParameterList([torch.nn.Parameter(torch.zeros(1))]), lr=death_rate)
        self.cosine_stepper = torch.optim.lr_scheduler.CosineAnnealingLR(self.sgd, T_max, eta_min, last_epoch)

    def step(self):
        self.cosine_stepper.step()

    def get_dr(self):
        return self.sgd.param_groups[0]['lr']

class LinearDecay(object):
    def __init__(self, death_rate, factor=0.99, frequency=600):
        self.factor = factor
        self.steps = 0
        self.frequency = frequency

    def step(self):
        self.steps += 1

    def get_dr(self, death_rate):
        if self.steps > 0 and self.steps % self.frequency == 0:
            return death_rate*self.factor
        else:
            return death_rate


class Masking(object):
    def __init__(self, G_optimizer=False, D_optimizer=False,  death_rate_decay=False, death_rate=0.3, death='magnitude', growth='momentum', redistribution='momentum', **config):
        growth_modes = ['random', 'momentum', 'momentum_neuron', 'gradient']
        if growth not in growth_modes:
            print('Growth mode: {0} not supported!'.format(growth))
            print('Supported modes are:', str(growth_modes))

        self.device = torch.device("cuda")
        self.growth_mode = growth
        self.death_mode = death
        self.redistribution_mode = redistribution
        self.death_rate_decay = death_rate_decay

        self.G_model = None
        self.D_model = None
        self.G_masks = {}
        self.D_masks = {}
        self.G_names = []
        self.D_names = []
        self.G_optimizer = G_optimizer
        self.D_optimizer = D_optimizer
        self.dy_mode = config['dy_mode']

        # stats
        self.G_name2zeros = {}
        self.G_num_remove = {}
        self.G_name2nonzeros = {}
        self.D_name2zeros = {}
        self.D_num_remove = {}
        self.D_name2nonzeros = {}
        self.death_rate = death_rate
        self.steps = 0


        # if fix, we do not explore the sparse connectivity
        if not self.dy_mode: self.prune_every_k_steps = None
        else: self.prune_every_k_steps = config['update_frequency']

    def init(self, mode='ERK', density=0.05, ratio_G=0.5, erk_power_scale=1.0):
        # calculating density for G and D3
        # some problems of defining sparsity ratio
        G_total_params = 0
        for name, weight in self.G_masks.items():
            G_total_params += weight.numel()

        D_total_params = 0
        for name, weight in self.D_masks.items():
            D_total_params += weight.numel()

        total_params = G_total_params + D_total_params

        self.G_density = (ratio_G * density * total_params) / G_total_params
        # what if G_density = 1.0?
        if self.G_density >= 1.0:
            self.G_density = 1.0
            self.D_density = (density * total_params - G_total_params) / D_total_params
        else:
            self.D_density = (total_params * density - G_total_params * self.G_density) / D_total_params


        if mode == 'uniform':
            for name, weight in self.G_model.named_parameters():
                if name not in self.G_masks: continue
                self.G_masks[name][:] = (torch.rand(weight.shape) < self.G_density).float().data.cuda()
            for name, weight in self.D_model.named_parameters():
                if name not in self.D_masks: continue
                self.D_masks[name][:] = (torch.rand(weight.shape) < self.D_density).float().data.cuda()

        elif mode == 'ERK':
            print('initialize by ERK')
            self.ERK_initialize_G(erk_power_scale)
            self.ERK_initialize_D(erk_power_scale)

        self.apply_mask(apply_mode='GD')
        self.G_fired_masks = copy.deepcopy(self.G_masks)  # used for ITOP
        self.D_fired_masks = copy.deepcopy(self.D_masks)  # used for ITOP

        for name, tensor in self.G_model.named_parameters():
            print(name, (tensor!=0).sum().item()/tensor.numel())

        G_total_size = 0
        G_sparse_size = 0
        for name, weight in self.G_masks.items():
            G_total_size += weight.numel()
            G_sparse_size += (weight != 0).sum().int().item()
        print('Total Model parameters of G:', G_total_size)
        print('Total parameters under sparsity level of {0}: {1}'.format(self.G_density, G_sparse_size / G_total_size))

        D_total_size = 0
        D_sparse_size = 0
        for name, weight in self.D_masks.items():
            D_total_size += weight.numel()
            D_sparse_size += (weight != 0).sum().int().item()
        print('Total Model parameters of D:', D_total_size)
        print('Total parameters under sparsity level of {0}: {1}'.format(self.D_density, D_sparse_size / D_total_size))

    def step(self, explore_mode='GD'):
        if 'G' in explore_mode:
            self.G_optimizer.step()
            self.apply_mask(apply_mode='G')

            self.death_rate_decay.step()
            self.death_rate = self.death_rate_decay.get_dr()
            self.steps += 1

        if 'D' in explore_mode:
            self.D_optimizer.step()
            self.apply_mask(apply_mode='D')

        if self.prune_every_k_steps is not None:
            if self.steps % self.prune_every_k_steps == 0 and self.steps > 0 and 'G' in explore_mode:
                self.truncate_weights(dy_mode=self.dy_mode)
                self.print_nonzero_counts(dy_mode=self.dy_mode)
                self.fired_masks_update()

    def add_module(self, G_model, D_model, ratio_G=0.5 , density=0.5, sparse_init='ERK'):
        self.G_model = G_model
        self.D_model = D_model

        for name, tensor in self.G_model.named_parameters():
            self.G_names.append(name)
            self.G_masks[name] = torch.zeros_like(tensor,  requires_grad=False).cuda()

        for name, tensor in self.D_model.named_parameters():
            self.D_names.append(name)
            self.D_masks[name] = torch.zeros_like(tensor,  requires_grad=False).cuda()

        print('Removing linear layer ...')
        self.remove_weight_partial_name('linear')
        print('Removing biases...')
        self.remove_weight_partial_name('bias')
        print('Removing gain...')
        self.remove_weight_partial_name('gain')
        print('Removing D.embed...')
        self.remove_weight_partial_name('D.embed')
        self.init(mode=sparse_init, density=density, ratio_G=ratio_G)


    def remove_weight(self, name):
        if name in self.masks:
            print('Removing {0} of size {1} = {2} parameters.'.format(name, self.masks[name].shape,
                                                                      self.masks[name].numel()))
            self.masks.pop(name)
        elif name + '.weight' in self.masks:
            print('Removing {0} of size {1} = {2} parameters.'.format(name, self.masks[name + '.weight'].shape,
                                                                      self.masks[name + '.weight'].numel()))
            self.masks.pop(name + '.weight')
        else:
            print('ERROR', name)

    def remove_weight_partial_name(self, partial_name):
        # removing G
        removed = set()
        for name in list(self.G_masks.keys()):
            if partial_name in name:

                print('Removing {0} of size {1} with {2} parameters...'.format(name, self.G_masks[name].shape,
                                                                                   np.prod(self.G_masks[name].shape)))
                removed.add(name)
                self.G_masks.pop(name)

        print('Removed {0} layers from Gegenerator.'.format(len(removed)))

        i = 0
        while i < len(self.G_names):
            name = self.G_names[i]
            if name in removed:
                self.G_names.pop(i)
            else:
                i += 1

        # removing D
        removed = set()
        for name in list(self.D_masks.keys()):
            if partial_name in name:
                print('Removing {0} of size {1} with {2} parameters...'.format(name, self.D_masks[name].shape,
                                                                               np.prod(self.D_masks[name].shape)))
                removed.add(name)
                self.D_masks.pop(name)

        print('Removed {0} layers from Discriminator.'.format(len(removed)))

        i = 0
        while i < len(self.D_names):
            name = self.D_names[i]
            if name in removed:
                self.D_names.pop(i)
            else:
                i += 1

    def remove_type(self, nn_type):
        for module in self.modules:
            for name, module in module.named_modules():
                if isinstance(module, nn_type):
                    self.remove_weight(name)

    def apply_mask(self, apply_mode='GD'):
        if 'G' in apply_mode:
            model_para = get_model_params(self.G_model)
            for name in model_para:
                if name in self.G_masks:
                    model_para[name] = model_para[name]*self.G_masks[name]
            set_model_params(self.G_model, model_para)

        if 'D' in apply_mode:
            model_para = get_model_params(self.D_model)
            for name in model_para:
                if name in self.D_masks:
                    model_para[name] = model_para[name] * self.D_masks[name]
            set_model_params(self.D_model, model_para)

    def truncate_weights(self, dy_mode):
        print(f"perform weight exploration for{dy_mode}")

        # update G
        if 'G' in dy_mode:
            # prune
            for name, weight in self.G_model.named_parameters():
                if name not in self.G_masks: continue
                mask = self.G_masks[name]
                self.G_name2nonzeros[name] = mask.sum().item()
                self.G_name2zeros[name] = mask.numel() - self.G_name2nonzeros[name]
                new_mask = self.magnitude_death(mask, weight, name, self.G_name2nonzeros[name], self.G_name2zeros[name])
                self.G_masks[name][:] = new_mask
            # grow
            for name, weight in self.G_model.named_parameters():
                if name not in self.G_masks: continue
                new_mask = self.G_masks[name].data.byte()
                num_remove = int(self.G_name2nonzeros[name] - new_mask.sum().item())

                # growth
                if self.growth_mode == 'random':
                    new_mask = self.random_growth(name, new_mask, weight, num_remove)

                if self.growth_mode == 'random_unfired':
                    new_mask = self.random_unfired_growth(name, new_mask, weight, num_remove)

                elif self.growth_mode == 'momentum':
                    new_mask = self.momentum_growth(name, new_mask, weight, num_remove)

                elif self.growth_mode == 'gradient':
                    new_mask = self.gradient_growth(name, new_mask, weight, num_remove)

                # exchanging masks
                self.G_masks.pop(name)
                self.G_masks[name] = new_mask.float()

        # update D
        if 'D' in dy_mode:
            # prune
            for name, weight in self.D_model.named_parameters():
                if name not in self.D_masks: continue
                mask = self.D_masks[name]
                self.D_name2nonzeros[name] = mask.sum().item()
                self.D_name2zeros[name] = mask.numel() - self.D_name2nonzeros[name]
                new_mask = self.magnitude_death(mask, weight, name, self.D_name2nonzeros[name], self.D_name2zeros[name])
                self.D_masks[name][:] = new_mask
            # grow
            for name, weight in self.D_model.named_parameters():
                if name not in self.D_masks: continue
                new_mask = self.D_masks[name].data.byte()
                num_remove = int(self.D_name2nonzeros[name] - new_mask.sum().item())

                # growth
                if self.growth_mode == 'random':
                    new_mask = self.random_growth(name, new_mask, weight, num_remove)

                if self.growth_mode == 'random_unfired':
                    new_mask = self.random_unfired_growth(name, new_mask, weight, num_remove)

                elif self.growth_mode == 'momentum':
                    new_mask = self.momentum_growth(name, new_mask, weight, num_remove)

                elif self.growth_mode == 'gradient':
                    new_mask = self.gradient_growth(name, new_mask, weight, num_remove)

                # exchanging masks
                self.D_masks.pop(name)
                self.D_masks[name] = new_mask.float()

        self.apply_mask(apply_mode='GD')
    '''
                    DEATH
    '''
    def threshold_death(self, mask, weight, name):
        return (torch.abs(weight.data) > self.threshold)

    def taylor_FO(self, mask, weight, name):

        num_remove = math.ceil(self.death_rate * self.name2nonzeros[name])
        num_zeros = self.name2zeros[name]
        k = math.ceil(num_zeros + num_remove)

        x, idx = torch.sort((weight.data * weight.grad).pow(2).flatten())
        mask.data.view(-1)[idx[:k]] = 0.0

        return mask

    def magnitude_death(self, mask, weight, name, num_nonzero, num_zero):

        num_remove = math.ceil(self.death_rate*num_nonzero)
        if num_remove == 0.0: return weight.data != 0.0

        x, idx = torch.sort(torch.abs(weight.data.view(-1)))
        n = idx.shape[0]

        k = math.ceil(num_zero + num_remove)
        threshold = x[k-1].item()

        return (torch.abs(weight.data) > threshold)


    def magnitude_and_negativity_death(self, mask, weight, name):
        num_remove = math.ceil(self.death_rate*self.name2nonzeros[name])
        num_zeros = self.name2zeros[name]

        # find magnitude threshold
        # remove all weights which absolute value is smaller than threshold
        x, idx = torch.sort(weight[weight > 0.0].data.view(-1))
        k = math.ceil(num_remove/2.0)
        if k >= x.shape[0]:
            k = x.shape[0]

        threshold_magnitude = x[k-1].item()

        # find negativity threshold
        # remove all weights which are smaller than threshold
        x, idx = torch.sort(weight[weight < 0.0].view(-1))
        k = math.ceil(num_remove/2.0)
        if k >= x.shape[0]:
            k = x.shape[0]
        threshold_negativity = x[k-1].item()


        pos_mask = (weight.data > threshold_magnitude) & (weight.data > 0.0)
        neg_mask = (weight.data < threshold_negativity) & (weight.data < 0.0)


        new_mask = pos_mask | neg_mask
        return new_mask

    '''
                    GROWTH
    '''

    def random_unfired_growth(self, name, new_mask, weight, num_remove):
        total_regrowth = num_remove
        n = (new_mask == 0).sum().item()
        if n == 0: return new_mask
        num_nonfired_weights = (self.fired_masks[name]==0).sum().item()

        if total_regrowth <= num_nonfired_weights:
            idx = (self.fired_masks[name].flatten() == 0).nonzero()
            indices = torch.randperm(len(idx))[:total_regrowth]

            # idx = torch.nonzero(self.fired_masks[name].flatten())
            new_mask.data.view(-1)[idx[indices]] = 1.0
        else:
            new_mask[self.fired_masks[name]==0] = 1.0
            n = (new_mask == 0).sum().item()
            expeced_growth_probability = ((total_regrowth-num_nonfired_weights) / n)
            new_weights = torch.rand(new_mask.shape).cuda() < expeced_growth_probability
            new_mask = new_mask.byte() | new_weights
        return new_mask

    def random_growth(self, name, new_mask, weight, num_remove):
        n = (new_mask==0).sum().item()
        if n == 0: return new_mask
        expeced_growth_probability = (num_remove/n)
        new_weights = torch.rand(new_mask.shape).cuda() < expeced_growth_probability
        new_mask_ = new_mask.bool() | new_weights
        if (new_mask_!=0).sum().item() == 0:
            new_mask_ = new_mask
        return new_mask_

    def momentum_growth(self, name, new_mask, weight, num_remove):
        grad = self.get_momentum_for_weight(weight)
        grad = grad*(new_mask==0).float()
        y, idx = torch.sort(torch.abs(grad).flatten(), descending=True)
        new_mask.data.view(-1)[idx[:num_remove]] = 1.0

        return new_mask

    def gradient_growth(self, name, new_mask, weight, num_remove):
        grad = self.get_gradient_for_weights(weight)
        grad = grad*(new_mask==0).float()

        y, idx = torch.sort(torch.abs(grad).flatten(), descending=True)
        new_mask.data.view(-1)[idx[:num_remove]] = 1.0

        return new_mask



    def momentum_neuron_growth(self, name, new_mask, weight, num_remove):
        total_regrowth = self.num_remove[name]
        grad = self.get_momentum_for_weight(weight)

        M = torch.abs(grad)
        if len(M.shape) == 2: sum_dim = [1]
        elif len(M.shape) == 4: sum_dim = [1, 2, 3]

        v = M.mean(sum_dim).data
        v /= v.sum()

        slots_per_neuron = (new_mask==0).sum(sum_dim)

        M = M*(new_mask==0).float()
        for i, fraction  in enumerate(v):
            neuron_regrowth = math.floor(fraction.item()*total_regrowth)
            available = slots_per_neuron[i].item()

            y, idx = torch.sort(M[i].flatten())
            if neuron_regrowth > available:
                neuron_regrowth = available
            threshold = y[-(neuron_regrowth)].item()
            if threshold == 0.0: continue
            if neuron_regrowth < 10: continue
            new_mask[i] = new_mask[i] | (M[i] > threshold)

        return new_mask

    '''
                UTILITY
    '''

    def gather_statistics(self):
        self.G_name2nonzeros = {}
        self.G_name2zeros = {}
        self.G_name2variance = {}

        self.G_total_variance = 0.0
        self.G_total_removed = 0
        self.G_total_nonzero = 0
        self.G_total_zero = 0.0

        self.D_name2nonzeros = {}
        self.D_name2zeros = {}
        self.D_name2variance = {}

        self.D_total_variance = 0.0
        self.D_total_removed = 0
        self.D_total_nonzero = 0
        self.D_total_zero = 0.0


        for name, tensor in self.G_model.named_parameters():
            if name not in self.G_masks: continue
            mask = self.G_masks[name]

            if self.redistribution_mode == 'gradient':
                grad = self.get_gradient_for_weights(tensor)
                self.G_name2variance[name] = torch.abs(grad[mask.byte()]).mean().item()  # /(V1val*V2val)
            elif self.redistribution_mode == 'magnitude':
                self.G_name2variance[name] = torch.abs(tensor)[mask.byte()].mean().item()
            elif self.redistribution_mode == 'none':
                self.G_name2variance[name] = 1.0
            elif self.redistribution_mode == 'uniform_distribution':
                self.G_name2variance[name] = 1
            else:
                print('Unknown redistribution mode:{0}'.format(self.redistribution_mode))
                raise Exception('Unknown redistribution mode!')

            if not np.isnan(self.G_name2variance[name]):
                self.G_total_variance += self.G_name2variance[name]
            self.G_name2nonzeros[name] = mask.sum().item()
            self.G_name2zeros[name] = mask.numel() - self.G_name2nonzeros[name]


            num_remove = math.ceil(self.death_rate * self.G_name2nonzeros[name])
            self.total_removed += num_remove
            self.total_nonzero += self.name2nonzeros[name]
            self.total_zero += self.name2zeros[name]


    def get_momentum_for_weight(self, weight):
        if 'exp_avg' in self.optimizer.state[weight]:
            adam_m1 = self.optimizer.state[weight]['exp_avg']
            adam_m2 = self.optimizer.state[weight]['exp_avg_sq']
            grad = adam_m1/(torch.sqrt(adam_m2) + 1e-08)
        elif 'momentum_buffer' in self.optimizer.state[weight]:
            grad = self.optimizer.state[weight]['momentum_buffer']
        return grad

    def get_gradient_for_weights(self, weight):
        grad = weight.grad.clone()
        return grad

    def print_nonzero_counts(self, dy_mode):
        if 'G' in dy_mode:
            for name, tensor in self.G_model.named_parameters():
                if name not in self.G_masks: continue
                mask = self.G_masks[name]
                num_nonzeros = (mask != 0).sum().item()
                val = '{0}: {1}->{2}, density: {3:.3f}'.format(name, self.G_name2nonzeros[name], num_nonzeros, num_nonzeros/float(mask.numel()))
                print(val)

        if 'D' in dy_mode:
            for name, tensor in self.D_model.named_parameters():
                if name not in self.D_masks: continue
                mask = self.D_masks[name]
                num_nonzeros = (mask != 0).sum().item()
                val = '{0}: {1}->{2}, density: {3:.3f}'.format(name, self.D_name2nonzeros[name], num_nonzeros, num_nonzeros/float(mask.numel()))
                print(val)
        print('Death rate: {0}\n'.format(self.death_rate))

    def fired_masks_update(self):
        G_ntotal_fired_weights = 0.0
        G_ntotal_weights = 0.0
        G_layer_fired_weights = {}
        for name, weight in self.G_model.named_parameters():
            if name not in self.G_masks: continue
            self.G_fired_masks[name] = self.G_masks[name].data.byte() | self.G_fired_masks[name].data.byte()
            G_ntotal_fired_weights += float(self.G_fired_masks[name].sum().item())
            G_ntotal_weights += float(self.G_fired_masks[name].numel())
            G_layer_fired_weights[name] = float(self.G_fired_masks[name].sum().item()) / float(
                self.G_fired_masks[name].numel())
            print('Layerwise percentage of the fired weights of', name, 'is:', G_layer_fired_weights[name])
        G_total_fired_weights = G_ntotal_fired_weights / G_ntotal_weights
        print('The percentage of the total fired weights of G is:', G_total_fired_weights)

        D_ntotal_fired_weights = 0.0
        D_ntotal_weights = 0.0
        D_layer_fired_weights = {}
        for name, weight in self.D_model.named_parameters():
            if name not in self.D_masks: continue
            self.D_fired_masks[name] = self.D_masks[name].data.byte() | self.D_fired_masks[name].data.byte()
            D_ntotal_fired_weights += float(self.D_fired_masks[name].sum().item())
            D_ntotal_weights += float(self.D_fired_masks[name].numel())
            D_layer_fired_weights[name] = float(self.D_fired_masks[name].sum().item()) / float(
                self.D_fired_masks[name].numel())
            print('Layerwise percentage of the fired weights of', name, 'is:', D_layer_fired_weights[name])
        D_total_fired_weights = D_ntotal_fired_weights / D_ntotal_weights
        print('The percentage of the total fired weights of D is:', D_total_fired_weights)


    def ERK_initialize_G(self, erk_power_scale):
        total_params = 0
        for name, weight in self.G_masks.items():
            total_params += weight.numel()
        is_epsilon_valid = False
        dense_layers = set()
        while not is_epsilon_valid:
            # eps * (p_1 * N_1 + p_2 * N_2) + (N_3 + N_4) =
            #    (1 - default_sparsity) * (N_1 + N_2 + N_3 + N_4)
            # eps * (p_1 * N_1 + p_2 * N_2) =
            #    (1 - default_sparsity) * (N_1 + N_2) - default_sparsity * (N_3 + N_4)
            # eps = rhs / (\sum_i p_i * N_i) = rhs / divisor.
            divisor = 0
            rhs = 0
            raw_probabilities = {}
            for name, mask in self.G_masks.items():
                n_param = np.prod(mask.shape)
                n_zeros = n_param * (1 - self.G_density)
                n_ones = n_param * self.G_density

                if name in dense_layers:
                    rhs -= n_zeros
                else:
                    rhs += n_ones
                    raw_probabilities[name] = (
                                                      np.sum(mask.shape) / np.prod(mask.shape)
                                              ) ** erk_power_scale
                    divisor += raw_probabilities[name] * n_param
            epsilon = rhs / divisor
            max_prob = np.max(list(raw_probabilities.values()))
            max_prob_one = max_prob * epsilon
            if max_prob_one > 1:
                is_epsilon_valid = False
                for mask_name, mask_raw_prob in raw_probabilities.items():
                    if mask_raw_prob == max_prob:
                        print(f"Sparsity of var:{mask_name} had to be set to 0.")
                        dense_layers.add(mask_name)
            else:
                is_epsilon_valid = True

        density_dict = {}
        total_nonzero = 0.0
        # With the valid epsilon, we can set sparsities of the remaning layers.
        for name, mask in self.G_masks.items():
            n_param = np.prod(mask.shape)
            if name in dense_layers:
                density_dict[name] = 1.0
            else:
                probability_one = epsilon * raw_probabilities[name]
                density_dict[name] = probability_one
            print(
                f"layer: {name}, shape: {mask.shape}, density: {density_dict[name]}"
            )
            self.G_masks[name][:] = (torch.rand(mask.shape) < density_dict[name]).float().data.cuda()

            total_nonzero += density_dict[name] * mask.numel()
        print(f"Overall sparsity of G {total_nonzero / total_params}")

    def ERK_initialize_D(self, erk_power_scale):
        total_params = 0
        for name, weight in self.D_masks.items():
            total_params += weight.numel()
        is_epsilon_valid = False
        dense_layers = set()
        while not is_epsilon_valid:
            # eps * (p_1 * N_1 + p_2 * N_2) + (N_3 + N_4) =
            #    (1 - default_sparsity) * (N_1 + N_2 + N_3 + N_4)
            # eps * (p_1 * N_1 + p_2 * N_2) =
            #    (1 - default_sparsity) * (N_1 + N_2) - default_sparsity * (N_3 + N_4)
            # eps = rhs / (\sum_i p_i * N_i) = rhs / divisor.
            divisor = 0
            rhs = 0
            raw_probabilities = {}
            for name, mask in self.D_masks.items():
                n_param = np.prod(mask.shape)
                n_zeros = n_param * (1 - self.D_density)
                n_ones = n_param * self.D_density

                if name in dense_layers:
                    rhs -= n_zeros
                else:
                    rhs += n_ones
                    raw_probabilities[name] = (
                                                      np.sum(mask.shape) / np.prod(mask.shape)
                                              ) ** erk_power_scale
                    divisor += raw_probabilities[name] * n_param
            epsilon = rhs / divisor
            max_prob = np.max(list(raw_probabilities.values()))
            max_prob_one = max_prob * epsilon
            if max_prob_one > 1:
                is_epsilon_valid = False
                for mask_name, mask_raw_prob in raw_probabilities.items():
                    if mask_raw_prob == max_prob:
                        print(f"Sparsity of var:{mask_name} had to be set to 0.")
                        dense_layers.add(mask_name)
            else:
                is_epsilon_valid = True

        density_dict = {}
        total_nonzero = 0.0
        # With the valid epsilon, we can set sparsities of the remaning layers.
        for name, mask in self.D_masks.items():
            n_param = np.prod(mask.shape)
            if name in dense_layers:
                density_dict[name] = 1.0
            else:
                probability_one = epsilon * raw_probabilities[name]
                density_dict[name] = probability_one
            print(
                f"layer: {name}, shape: {mask.shape}, density: {density_dict[name]}"
            )
            self.D_masks[name][:] = (torch.rand(mask.shape) < density_dict[name]).float().data.cuda()

            total_nonzero += density_dict[name] * mask.numel()
        print(f"Overall sparsity of D {total_nonzero / total_params}")

