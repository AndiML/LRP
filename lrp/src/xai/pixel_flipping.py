import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, datasets, models
from torchvision.transforms import v2
import numpy as np
import sys
import argparse
import os
from pathlib import Path
from argparse import ArgumentParser
from scipy.stats import pearsonr
from matplotlib import pyplot as plt
from functools import partial
import pandas as pd


try:
    ON_CLUSTER = os.environ["ON_CLUSTER"] == "True"
except:
    ON_CLUSTER = False

if ON_CLUSTER:
    sys.path.append("/users/manuelwe/scripts/python")

from models import get_model
from utils import evaluate, get_data_loaders, integrated_gradients, gradient, AddInverse, viz_expl, shrink_classifier

from ImageInpainting.model import PConvUNet as Inpainter

torch.manual_seed(0)

def test_localization(model, samples, detach_ctx = False, n_iter = 1, gridsize = 3):
    
    done = False
    block_size = samples[0].shape[2]

    correct = 0
    total = 0

    while not done:

        classes = np.random.choice(len(samples), size = gridsize**2, replace = False)

        cl_idx = 0
        
        grid = []
        cl_2_loc = {}

        for row in range(gridsize):
            
            grid_row = []

            for col in range(gridsize):
                
                cl = classes[cl_idx]
                cl_idx += 1
                
                cl_2_loc[cl] = (row, col)

                sample_idx = np.random.choice(len(samples[cl]), size = 1)
                img = samples[cl][sample_idx]
                samples[cl] = samples[cl][:sample_idx] + samples[cl][sample_idx + 1:]

                # img = samples[cl][0]
                # samples[cl] = samples[cl][1:]

                if len(samples[cl]) == 0:
                    done = True

                grid_row += [img]

            grid_row = torch.cat(grid_row, axis = 2)
            grid += [grid_row]

        grid_viz = torch.cat(grid, axis = 1)
        grid = torch.cat(grid, axis = 1).unsqueeze(axis = 0).repeat(gridsize ** 2, 1, 1, 1)

        expl = integrated_gradients(grid, model, torch.tensor(classes), n_iter = n_iter, detach_ctx = detach_ctx)
        expl = expl.sum(axis = -1)

        for cl_idx, cl in enumerate(classes):

            row, col = cl_2_loc[cl]

            # Sum up the contribution of each image on the grid
            expl_pooled = F.avg_pool2d(
                torch.tensor(expl[cl_idx]).clamp(min = 0).view(1, 1,*expl[cl_idx].shape),
                kernel_size = block_size,
                stride = block_size
            ).view(gridsize, gridsize)

            total += 1

            if cl_idx == expl_pooled.argmax():
                correct += 1
            
            print()

            print(f"predicted = {expl_pooled.argmax()}")
            print(f"ground-truth = {cl_idx}")
            print(cl)
            fig, axs = plt.subplots(2)
            axs[0].imshow(grid_viz[:3].permute(1,2,0).numpy()) 
            viz_expl(axs[1], expl[cl_idx][None,:], 1, colorbar = False)

            plt.show()
            
            
    print(f"Localization accuracy = {correct / total}")
    return correct / total

def get_n_sensitvity_blockified(model, batch, block_size, expl_postprocessing, grad_steps = 1, repeats = 300, baseline = [0,0,0]):

    data, targets = batch
    model = model.eval().to(DEVICE)

    data, targets = batch
    logits_initial = model(data.to(DEVICE)).detach().cpu()
    
    output_size = logits_initial.shape[1]
    input_size = data.shape[2]
    input_channels = data.shape[1]

    # new_input_size = input_size // block_size

    if data.shape[3] != input_size:
        raise ValueError("only images of squared size are supported")
    
    logits_initial = logits_initial[torch.eye(output_size)[targets].bool()]
    expl = torch.tensor(apply_postprocessing(expl_postprocessing, integrated_gradients(data, model, targets, n_iter = grad_steps, detach_ctx = False)))
    expl_det = torch.tensor(apply_postprocessing(expl_postprocessing, integrated_gradients(data, model, targets, n_iter = grad_steps, detach_ctx = True)))

    # expl = F.avg_pool2d(
    #     torch.tensor(expl)[:, None, :],
    #     kernel_size = block_size,
    #     stride = block_size
    # ).view(-1, new_input_size, new_input_size)

    drops = []
    drops_predicted = []
    drops_predicted_det = []

    # print(expl.shape)

    import random

    cells = []
    
    for row in range(input_size):
        for col in range(input_size):
            
            if row + block_size <= input_size and col + block_size <= input_size:
                cells += [(row, col)]

    for _ in range(repeats):

        perturbed = data.clone().permute(0,2,3,1)
        # print(perturbed.shape)
        expl_mass = []
        expl_mass_det = []

        for ind, input in enumerate(perturbed):
            
            cell = random.choice(cells)
            # print(cell)

            if input_channels == 6:
                # input[cell[0] : cell[0] + block_size, cell[1] : cell[1] + block_size] = torch.cat([torch.tensor(baseline), 1 - torch.tensor(baseline)]).float()
                input[cell[0] : cell[0] + block_size, cell[1] : cell[1] + block_size] = torch.cat([torch.tensor(baseline), torch.tensor(baseline)]).float()
            else: 
                input[cell[0] : cell[0] + block_size, cell[1] : cell[1] + block_size] = torch.tensor(baseline).float()

            # drops_predicted += [torch.where(mask, expl, 0).sum(axis = (1,2))]
            # print(expl[ind].shape)
            # print(expl[ind, cell[0] : cell[0] + block_size, cell[1] : cell[1] + block_size].shape)

            expl_mass += [expl[ind, cell[0] : cell[0] + block_size, cell[1] : cell[1] + block_size].sum()]
            expl_mass_det += [expl_det[ind, cell[0] : cell[0] + block_size, cell[1] : cell[1] + block_size].sum()]
        
        drops_predicted += [torch.tensor(expl_mass)]
        drops_predicted_det += [torch.tensor(expl_mass_det)]

        perturbed = perturbed.permute(0, 3, 1, 2)
        # print(perturbed.shape)


        # pixel_indices = np.random.choice(new_input_size**2, size = len(data), replace = True)
        # mask = torch.eye(new_input_size**2)[pixel_indices].view(-1, 1, new_input_size, new_input_size)
        
        # # Turn superpixel mask into pixel mask
        # # it_mask = F.upsample(it_mask, scale_factor = (block_size, block_size), mode = "nearest").bool().view(-1, 1, input_size ** 2).repeat(1, input_channels, 1)
        # mask = F.upsample(mask, scale_factor = (block_size, block_size), mode = "nearest").bool().view(-1, input_size, input_size)#.repeat(1, input_channels, 1)
        
        # perturbed = data.clone()

        # perturbed = perturbed.permute(0, 2, 3, 1)

        # if input_channels == 6:
        #     perturbed[mask] = torch.cat([torch.tensor(baseline), 1 - torch.tensor(baseline)]).float()
        # else: 
        #     perturbed[mask] = torch.tensor(baseline).float()

        # perturbed = perturbed.permute(0, 3, 1, 2)
        # from matplotlib import pyplot as plt

        # viz = np.concatenate([img for img in perturbed.permute(0, 2, 3, 1).detach().cpu().numpy()[:5]], axis = 1)
        # plt.imshow(viz)
        # plt.show()

        logits = model(perturbed.to(DEVICE))

        drops += [logits_initial - logits.detach().cpu()[torch.eye(output_size)[targets].bool()]]

        # drops_predicted += [torch.where(mask, expl, 0).sum(axis = (1,2))]

    drops = torch.stack(drops).T
    drops_predicted = torch.stack(drops_predicted).T
    drops_predicted_det = torch.stack(drops_predicted_det).T
    
    return pearsonr(drops, drops_predicted, axis = 1).statistic.mean(), pearsonr(drops, drops_predicted_det, axis = 1).statistic.mean()

def get_n_sensitivity(model, batch, n_features, expl_postprocessing, grad_steps = 1, detach_ctx = False, repeats = 300, baseline = [0, 0, 0]):

    data, targets = batch
    model = model.eval().to(DEVICE)

    data, targets = batch
    logits_initial = model(data.to(DEVICE)).detach().cpu()
    
    output_size = logits_initial.shape[1]
    input_size = data.shape[2]
    input_channels = data.shape[1]

    if data.shape[3] != input_size:
        raise ValueError("only images of squared size are supported")
    
    logits_initial = logits_initial[torch.eye(output_size)[targets].bool()]

    expl = torch.tensor(apply_postprocessing(expl_postprocessing, integrated_gradients(data, model, targets, n_iter = grad_steps, detach_ctx = detach_ctx)).reshape(len(data), -1))
    drops = []
    drops_predicted = []

    for i in range(repeats):

        pixel_indices = np.stack([np.random.choice(input_size**2, size = n_features, replace = False) for _ in range(len(data))])
        
        mask = torch.eye(input_size**2)[pixel_indices].sum(axis = 1).bool()

        perturbed = data.clone().view(len(data), input_channels, -1)

        perturbed = perturbed.permute(0, 2, 1)

        if input_channels == 6:
            perturbed[mask] = torch.cat([torch.tensor(baseline), 1 - torch.tensor(baseline)]).float()
        else: 
            perturbed[mask] = torch.tensor(baseline).float()

        perturbed = perturbed.permute(0, 2, 1).view(-1, input_channels, input_size, input_size)
        
        # perturbed[mask[:,None,:].repeat(1,3,1)] = 0
        # perturbed = perturbed.view(len(data), 3, input_size, input_size)

        logits = model(perturbed.to(DEVICE))

        drops += [logits_initial - logits.detach().cpu()[torch.eye(output_size)[targets].bool()]]
        drops_predicted += [torch.where(mask, expl, 0).sum(axis = -1)]

    drops = torch.stack(drops).T
    drops_predicted = torch.stack(drops_predicted).T
    
    return pearsonr(drops, drops_predicted, axis = 1).statistic.mean()

def get_perturbation_curve(model, batch, expl, block_size = 8, order = "morf", inpainter = None, random = False, baseline = [0, 0, 0], add = False):

    # expl is a numpy array of shape (batch_size, input_size, input_size)

    if order not in ["lerf", "morf"]:
        raise ValueError("invalid 'order' argument")
    
    data, targets = batch
    
    input_size = data.shape[2] 
    input_channels = data.shape[1]

    if data.shape[3] != input_size:
        raise ValueError("only images of squared size are supported")

    model = model.eval().to(DEVICE)
    
    if inpainter is not None:
        inpainter.eval()
        
    N = len(data)
    
    new_input_size = input_size // block_size
    
    # Pool relevance into blocks of size (block_size x block_size)
    expl = F.avg_pool2d(
        torch.tensor(expl)[:, None, :],
        kernel_size = block_size,
        stride = block_size
    ).view(-1, new_input_size, new_input_size)
    
    # Flatten superpixels and sort by relevance in descending order
    expl_flat = expl.reshape((N, -1))
    inp_flat = data.view(N, input_channels, -1)    
    sorted_positions = np.argsort(expl_flat, axis = 1)
    
    if random:
        sorted_positions = sorted_positions.numpy().T
        np.random.shuffle(sorted_positions)
        sorted_positions = torch.tensor(sorted_positions.T)        

    if order == "lerf": 
        sorted_positions = sorted_positions.flip(dims = (1,))

    if add:
        # initial_out = F.softmax(model(torch.zeros_like(data).to(DEVICE)), dim = 1).detach().cpu()
        initial_out = model(torch.zeros_like(data).to(DEVICE)).detach().cpu()

    else:
        # initial_out = F.softmax(model(data.to(DEVICE)), dim = 1).detach().cpu()
        initial_out = model(data.to(DEVICE)).detach().cpu()


    output_size = initial_out.shape[1]

    probs = [initial_out[torch.eye(output_size)[targets].bool()]]
    
    # In each iteration i, the mask will be used to index the i most/least relevant pixel blocks
    mask = torch.zeros((inp_flat.shape[0], inp_flat.shape[2])).bool()

    perturbation_steps = np.arange(1, new_input_size + 1) ** 2
    
    for i in range(1, expl_flat.shape[1] + 1):
        
        max_rel_ind = sorted_positions[:, -i]
        
        # Create mask for this iteration's pixel block
        it_mask = torch.eye(expl_flat.shape[1])[max_rel_ind].view(-1, 1, new_input_size, new_input_size)
        
        # Turn superpixel mask into pixel mask
        it_mask = F.upsample(it_mask, scale_factor = (block_size, block_size), mode = "nearest").bool().view(-1, input_size ** 2)#.repeat(1, input_channels, 1)
        
        # Add this iteration's mask to the overall mask
        new_mask = mask | it_mask
        
        # Sanity check
        # assert((new_mask.int() - mask.int()).sum() == N * input_channels * block_size ** 2)
        assert((new_mask.int() - mask.int()).sum() == N * block_size ** 2)
        
        mask = new_mask

        if i in perturbation_steps:
            
            if inpainter is None:

                perturbed_input = inp_flat.clone().permute(0, 2, 1)
                
                if add:
                    block_mask = ~mask
                else:
                    block_mask = mask

                if input_channels == 6:
                    # Could also just set it all to 0?
                    perturbed_input[block_mask] = torch.cat([torch.tensor(baseline), 1 - torch.tensor(baseline)]).float()
                else:
                    perturbed_input[block_mask] = torch.tensor(baseline).float()

                perturbed_input = perturbed_input.permute(0, 2, 1)

            else:
                raise NotImplementedError("Inpainter functionality needs a rework")
                # One input to the inpainter is the inverted mask
                mask_2d = (~mask.reshape(-1, input_channels, input_size, input_size)).float()

                with torch.no_grad():
                    
                    rec, _ = inpainter((data.clone() * mask_2d).to(DEVICE), mask_2d.to(DEVICE))

                    rec = rec.view(N, input_channels, -1).detach().cpu()

                    perturbed_input[mask] = rec[mask]

            inp = perturbed_input.view(N, input_channels, input_size, input_size)

            out = model(inp.to(DEVICE))
            # probs += [F.softmax(out, dim = 1).detach().cpu()[torch.eye(output_size)[targets].bool()]]
            probs += [out.detach().cpu()[torch.eye(output_size)[targets].bool()]]

    probs = torch.stack(probs).numpy()
    
    return probs

def get_dataset_curves(model, expl_postprocessing, block_size, create_loader_func, ref_model = None, n_iter = 1, lrp_gamma = 0.0, detach_ctx = False, random = False, inpainter = None, baseline = [0, 0, 0], n_samples = 5000, seed = 0, extra_transform = None, gamma_lim = -1, add_lerf = False, teacher_norm = False, teacher = None):
    
    assert not (teacher_norm and teacher == None)

    model = model.eval().to(DEVICE)
    morf_curve = []
    lerf_curve = []

    processed_samples = 0

    if ref_model is not None:
        eval_model = ref_model
    else:
        eval_model = model

    torch.manual_seed(seed)
    _, loader = create_loader_func()
    
    for i, (data, targets) in enumerate(loader):
    
        if processed_samples >= n_samples:
            break
        
        extra_transform = extra_transform if extra_transform is not None else lambda x : x

        if hasattr(loader, "label_dict") and loader.label_dict is not None:
            targets = torch.tensor([loader.label_dict[t.item()] for t in targets])

        if n_iter <= 1:

            if teacher_norm:

                def hook(self, input, output):
                    self.output = output
                
                hooks = []

                hooks += [teacher.stem.proj.register_forward_hook(hook)]

                for i, block in enumerate(teacher.blocks):

                    hooks += [block.mlp_channels.act.register_forward_hook(hook)]
                    hooks += [block.mlp_channels.fc2.register_forward_hook(hook)]

                    if False and hasattr(model.blocks[i], "ctx_norm"):
                        hooks += [block.linear_tokens.register_forward_hook(hook)]

                teacher(data.to(DEVICE))
                
                norm_func = partial(torch.norm, p = 2, dim = -1, keepdim = True)
                model.stem.ctx.fixed_norm = norm_func(teacher.stem.proj.output.flatten(2).transpose(1, 2))

                for i, block in enumerate(teacher.blocks):
                    model.blocks[i].mlp_channels.ctx1.fixed_norm = norm_func(block.mlp_channels.act.output)
                    model.blocks[i].mlp_channels.ctx2.fixed_norm = norm_func(block.mlp_channels.fc2.output)
                    
                    if False and hasattr(model.blocks[i], "ctx_norm"):
                        model.blocks[i].ctx_norm.fixed_norm = norm_func(block.linear_tokens.output.transpose(1,2))

                for h in hooks:
                    h.remove()

            expl = model.forward_lrp(extra_transform(data), targets, gamma = lrp_gamma, dev = DEVICE, gamma_lim = gamma_lim)[1]
        else:
            print(f"using Integrated Gradients with {n_iter} grad steps")
            expl = integrated_gradients(extra_transform(data), model = model, targets = targets, n_iter = n_iter, detach_ctx = detach_ctx)

        expl = apply_postprocessing(expl_postprocessing, expl)
        get_pert_curve_func = partial(get_perturbation_curve, eval_model, (data, targets), expl, block_size = block_size, inpainter = inpainter, random = random, baseline = baseline)
        
        morf_curve += [get_pert_curve_func(order = "morf")]
        lerf_curve += [get_pert_curve_func(order = "lerf", add = add_lerf)]

        processed_samples += len(data)

    lerf_curve = np.concatenate(lerf_curve[:n_samples], axis = 1)
    morf_curve = np.concatenate(morf_curve[:n_samples], axis = 1)
    
    print("Standard errors:")

    print(lerf_curve.std(axis = 1) / n_samples**.5)
    print(morf_curve.std(axis = 1) / n_samples**.5)

    return morf_curve.mean(axis = 1), lerf_curve.mean(axis = 1)

def apply_postprocessing(functions, x):

    for f in functions:
        x = f(x)
    
    return x 

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("-p", "--p", type=float, default=0.0)
    parser.add_argument("-lrp_g", "--lrp_gamma", type=float, default=0.0)
    parser.add_argument("-flo", "--flip_original", action='store_true', default = False)
    parser.add_argument("-uncn", "--uncentered_norm", action='store_true', default = False)
    parser.add_argument("-lp", "--lipschitz", action='store_true', default=False)
    parser.add_argument("-abs", "--absolute", action='store_true', default=False)
    parser.add_argument("-cn", "--clamp_negative", action='store_true', default=False)
    parser.add_argument("-ph", "--posthoc", action='store_true', default=False)
    parser.add_argument("-inp", "--use_inpainter", action='store_true', default=False)
    parser.add_argument("-rnd", "--random", action='store_true', default=False)
    parser.add_argument("-cs", "--compute_sensitivity", action='store_true', default=False)
    parser.add_argument("-ub", "--use_biases", action='store_true', default = False)
    parser.add_argument("-bst", "--big_stem", action='store_true', default = False)
    parser.add_argument("-addl", "--add_lerf", action='store_true', default = False)
    parser.add_argument("-unl_g", "--unlim_gamma", action='store_true', default = False)
    parser.add_argument("-t_norm", "--teacher_norm", action='store_true', default = False)
    parser.add_argument("-alt", "--alternate", action='store_true', default = False)
    parser.add_argument("-dbg", "--debug", action='store_true', default = False)

    parser.add_argument("-fr_lim", "--freeze_lim", type=int, default=0)
    parser.add_argument("-b_lim", "--bias_lim", type=int, default=0)

    parser.add_argument("-bs", "--batch_size", type=int, default=200)
    parser.add_argument("-r", "--repeats", type=int, default=300)
    parser.add_argument("-ch", "--input_channels", type=int, default=3)
    parser.add_argument("-gs", "--grad_steps", type=int, default=1)
    parser.add_argument("-ns", "--n_samples", type=int, default=5000)
    parser.add_argument("-nl", "--n_layers", type=int, default=8)
    parser.add_argument("-ds", "--data_seed", type=int, default=0)
    parser.add_argument("-path", "--path", type=str, default = "")
    parser.add_argument("-d", "--dataset", type=str, default="CIFAR10")
    parser.add_argument("-m", "--model", type=str, default="basic")
    parser.add_argument("-a", "--activation", type=str, default="relu")
    parser.add_argument("-ctx", "--ctx_type", type=str, default="comp_conservative")
    parser.add_argument("-subt", "--sub_task", type=str, default="")
    parser.add_argument("-suf", "--suffix", type=str, default="")
    parser.add_argument("-rp", "--ref_path", type=str, default="")

    return parser.parse_args()

def main():
    
    args = parse_args().__dict__
    print(args)
    
    model_name = args["model"]
    lrp_gamma = args["lrp_gamma"]

    grad_steps = args["grad_steps"]
    assert not (lrp_gamma > 0 and grad_steps > 1)

    debug = args["debug"]
    input_channels = args["input_channels"]
    uncentered_norm = args["uncentered_norm"]
    p = args["p"]
    dataset = args["dataset"]
    random = args["random"]
    use_inpainter = args["use_inpainter"]
    lipschitz = args["lipschitz"]
    flip_original = args["flip_original"]
    ref_path = args["ref_path"]
    sub_task = args["sub_task"]
    input_size_dict = {
        "CIFAR10"  : 32,
        "CELEBA"   : 64,
        "IMGNET64" : 64,
        "TIN"      : 64,
        "ISIC"     : 224,
        "PCAM"     : 96,
        "IMGNET"   : 224
    }

    k_dict = {
        "CIFAR10"  : 1,
        "CELEBA"   : 1,
        "IMGNET64" : 3,
        "TIN"      : 2,
        "ISIC"     : 1,
        "PCAM"     : 1,
        "IMGNET"   : 1
    }

    output_size_dict = {
        "TIN" : 200,
        "CIFAR10" : 10,
        "CELEBA": 2,
        "ISIC" : 2,
        "PCAM" : 2,
        "IMGNET64" : 1000,
        "IMGNET"  : 1000
    }

    block_size_dict = {
        "CIFAR10" : 4,
        "CELEBA"  : 8,
        "IMGNET64": 8,
        "TIN" : 8,
        "ISIC" : 28,
        "PCAM" : 12,
        "IMGNET" : 16
    }

    scale_dict = {
        "CIFAR10" : 1,
        "CELEBA"  : 1,
        "IMGNET64": 100,
        "TIN" : 100,
        "ISIC" : 1,
        "PCAM" : 1,
        "IMGNET" : 1
    }

    loss_functions = {}
  
    loss_functions["comp_conservative"] = nn.CrossEntropyLoss()
    loss_functions["comp_decay"] = nn.CrossEntropyLoss()
    loss_functions["bcos"] = nn.CrossEntropyLoss()
    loss_functions["identity"] = nn.CrossEntropyLoss()

    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    post_augment_transforms = [v2.ToTensor(), v2.Normalize(mean, std)]

    input_size = input_size_dict[dataset]
    output_size = output_size_dict[dataset]
    block_size = block_size_dict[dataset]

    data_seed = args["data_seed"]
    posthoc = args["posthoc"]
    grad_steps = args["grad_steps"]
    abs = args["absolute"]
    clamp_negative = args["clamp_negative"]
    n_samples = args["n_samples"]
    big_stem = args["big_stem"]

    if  sub_task == "butterfly":
        remap_ind = True
        classes = list(range(321, 327))
        
    else:
        remap_ind = False
        classes = None

    fc_name = "fc" if model_name in ["resnet18", "resnet50"] else "head"

    create_loader = partial(get_data_loaders, dataset, post_augment_transforms, input_size = input_size,on_cluster = ON_CLUSTER, batch_size_test=args["batch_size"], shuffle_test=True, resize_crop = dataset == "IMGNET", test_only = True, remap_ind = remap_ind, test_classes = classes)
    
    torch.manual_seed(data_seed)
    _, val_loader = create_loader()

    if flip_original:
        
        get_ref_model_func = partial(get_model, model_name, 0.0, input_size, output_size, input_channels = 3, use_biases = True)

        if model_name in ["resnet18", "resnet50"]:
            ref_model = get_ref_model_func(weights_path = "torchvision")

        elif model_name in ["resmlp_12", "resmlp_24"]:
            ref_model = get_ref_model_func(weights_path = "timm", big_stem = False)

        elif model_name == "basic":
            ref_model = get_ref_model_func(weights_path = ref_path)

        else:
            raise NotImplementedError()
        
        ref_model.make_explainable(avgpool = False)

        if classes is not None:
            shrink_classifier(ref_model, classes, use_bias = True, fc_name = fc_name)

        ref_model = ref_model.eval().to(DEVICE)

        if not debug:
            print("Validating ref model")
            evaluate(ref_model, val_loader, loss_functions[args["ctx_type"]], compute_f1 = dataset in ["CELEBA", "ISIC"])

    else:
        ref_model = None

    freeze_lim = args["freeze_lim"]
    bias_lim = args["bias_lim"]

    model = get_model(model_name, p, input_size, output_size, num_layers = args["n_layers"], activation = args["activation"], posthoc_detachment = posthoc, context = args["ctx_type"], vgg = dataset == "ISIC", k = k_dict[dataset], scale = scale_dict[dataset], input_channels = input_channels, unc_norm = uncentered_norm, use_biases = args["use_biases"], big_stem = big_stem, freeze_lim = freeze_lim, bias_lim = bias_lim, alternate = args["alternate"])
    
    if lipschitz:
        model.make_lipschitz()

    if ON_CLUSTER:
        inpainters_dir = "/home/manuelwe/inpainters"
        curves_dir = "/home/manuelwe/curves"

    else:
        inpainters_dir = "../inpainters"
        curves_dir = "../curves"

    if input_channels == 6 and model_name in ["resnet18", "resnet50"]:
        old_weight = model.conv1.weight.data
        model.conv1 = nn.Conv2d(6, model.conv1.weight.shape[0],  model.conv1.kernel_size,  model.conv1.stride, model.conv1.padding, bias = False)
        model.conv1.weight.data[:, :3] = old_weight

    if input_channels == 6 and model_name in ["resmlp_12", "resmlp_24"]:
        old_weight = model.stem.proj.weight.data
        model.stem.proj = nn.Conv2d(6, model.stem.proj.weight.shape[0],  model.stem.proj.kernel_size,  model.stem.proj.stride, model.stem.proj.padding, bias = False)
        model.stem.proj.weight.data[:, :3] = old_weight

    if classes is not None:
        
        shrink_first = (args["path"] not in ["", "torchvision", "timm"]) and torch.load(args["path"], map_location=torch.device('cpu'))[f"{fc_name}.weight"].shape[0] == len(classes)
        if shrink_first:
            shrink_classifier(model, classes, fc_name = fc_name, use_bias = False)
        if args["path"] not in ["", "torchvision", "timm"]:
            model.load_state_dict(torch.load(args["path"], map_location=torch.device('cpu')))
        if not shrink_first:
            shrink_classifier(model, classes, fc_name = fc_name, use_bias = False)
            
    elif args["path"] not in ["", "torchvision", "timm"]:
        model.load_state_dict(torch.load(args["path"], map_location=torch.device('cpu')))

    use_avgpool = args["path"] not in ["", "torchvision"]

    model.make_explainable(avgpool = use_avgpool)

    extra_transform = (lambda x : AddInverse()(x, dim = 1)) if input_channels == 6 else None

    model = model.to(DEVICE).eval()
    
    if args["teacher_norm"]:
        import timm
        teacher = timm.create_model(f'resmlp_12_224.fb_in1k', pretrained=True).to(DEVICE).eval()
    else:
        teacher = None

    evaluate(model, val_loader, loss_functions[args["ctx_type"]], compute_f1 = dataset in ["CELEBA", "ISIC"], extra_transform = extra_transform, teacher = teacher, teacher_norm=args["teacher_norm"])

    if use_inpainter:

        inpainter = Inpainter()
        inpainter.load_state_dict(torch.load(f"{inpainters_dir}/{dataset.lower()}/unet_pconv", map_location=torch.device('cpu')))
        inpainter = inpainter.to(DEVICE)
        inpainter.eval()

    else:
        
        inpainter = None

    baseline = [0, 0, 0]

    morf = {}
    lerf = {}

    expl_postprocessing = [lambda expl : expl.sum(axis = -1)]

    if abs:
        expl_postprocessing = [lambda expl : np.abs(expl)] + expl_postprocessing
    
    if clamp_negative:
        expl_postprocessing += [lambda expl : np.clip(expl, min = 0)]

    dataset_path = f"{curves_dir}/{dataset.lower()}"
    Path(dataset_path).mkdir(parents=True, exist_ok=True)

    get_curve_func = partial(get_dataset_curves, model, expl_postprocessing, block_size, create_loader, ref_model = ref_model, n_iter = grad_steps, lrp_gamma = lrp_gamma, inpainter = inpainter, baseline = baseline, n_samples = n_samples, seed = data_seed, extra_transform = extra_transform, gamma_lim = 0 if args["unlim_gamma"] else freeze_lim, add_lerf = args["add_lerf"], teacher_norm = args["teacher_norm"], teacher = teacher)
    
    torch.manual_seed(data_seed)
    morf_curve, lerf_curve = get_curve_func(detach_ctx = False)
    
    morf[f"with_ctx"] = (morf_curve / morf_curve[0]).tolist()
    lerf[f"with_ctx"] = (lerf_curve / morf_curve[0]).tolist()
    
    torch.manual_seed(data_seed)
    morf_curve_dt, lerf_curve_dt = get_curve_func(detach_ctx = True)

    morf[f"detached_ctx"] = (morf_curve_dt / morf_curve_dt[0]).tolist()
    lerf[f"detached_ctx"] = (lerf_curve_dt / morf_curve_dt[0]).tolist()

    if random:

        torch.manual_seed(data_seed)
        morf_curve_rand, lerf_curve_rand = get_curve_func(random = True)

        morf[f"random"] = (morf_curve_rand / morf_curve_rand[0]).tolist()
        lerf[f"random"] = (lerf_curve_rand / morf_curve_rand[0]).tolist()

    df_lerf = pd.DataFrame.from_dict(lerf, orient = "index")
    df_morf = pd.DataFrame.from_dict(morf, orient = "index")

    lerf_path = f"{dataset_path}/lerf_p{p}_gs{grad_steps}_{args['model']}_{args['ctx_type']}_gam{lrp_gamma}_ch{input_channels}_ds{data_seed}_ns{n_samples}"
    morf_path = f"{dataset_path}/morf_p{p}_gs{grad_steps}_{args['model']}_{args['ctx_type']}_gam{lrp_gamma}_ch{input_channels}_ds{data_seed}_ns{n_samples}"

    if args['uncentered_norm']:
        lerf_path += "_uncn"
        morf_path += "_uncn"

    if use_inpainter:
        lerf_path += "_inp"
        morf_path += "_inp"

    if posthoc:
        lerf_path += "_posthoc"
        morf_path += "_posthoc"

    if abs:
        lerf_path += "_abs"
        morf_path += "_abs"

    if clamp_negative:
        lerf_path += "_cn"
        morf_path += "_cn"

    if lipschitz:
        lerf_path += "_lip"
        morf_path += "_lip"

    if model_name in ["mlp_mixer", "mlp_gating"]:
        lerf_path += f"_nl{args['n_layers']}"
        morf_path += f"_nl{args['n_layers']}"

    if flip_original:
        lerf_path += "_flo"
        morf_path += "_flo"
    
    if sub_task:
        lerf_path += f"_{sub_task}"
        morf_path += f"_{sub_task}"

    if freeze_lim != 0:
        lerf_path += f"_fr{freeze_lim}"
        morf_path += f"_fr{freeze_lim}"

    if bias_lim != 0:
        lerf_path += f"_b{bias_lim}"
        morf_path += f"_b{bias_lim}"
        
    if args["add_lerf"]:
        lerf_path += f"_add"
        morf_path += f"_add"

    if args["unlim_gamma"]:
        lerf_path += f"_unlg"
        morf_path += f"_unlg"
    
    if args["use_biases"]:
        lerf_path += f"_bias"
        morf_path += f"_bias"

    if args["teacher_norm"]:
        lerf_path += f"_tnorm"
        morf_path += f"_tnorm"

    if args["alternate"]:
        lerf_path += f"_alt"
        morf_path += f"_alt"
        
    lerf_path += args["suffix"]
    morf_path += args["suffix"]

    df_lerf.to_pickle(lerf_path)
    df_morf.to_pickle(morf_path)

    df_lerf.to_csv(lerf_path + ".csv")
    df_morf.to_csv(morf_path + ".csv")

    print(df_lerf)
    print(df_morf)

    print(f"lerf auc = {df_lerf.iloc[1].sum()}")
    print(f"morf auc = {df_morf.iloc[1].sum()}")

    if args["add_lerf"]:
        print(f"add auc = {(df_lerf.iloc[1] + df_morf.iloc[1]).sum()}")
    else:
        print(f"diff auc = {(df_lerf.iloc[1] - df_morf.iloc[1]).sum()}")


    if args["compute_sensitivity"]:

        _, val_loader = get_data_loaders(dataset, post_augment_transforms[args["model"]], input_size = input_size,on_cluster = ON_CLUSTER, batch_size_test=args["batch_size"], shuffle_test=True, resize_crop = dataset == "IMGNET", test_only = True)
        batch = next(iter(val_loader))

        sens_n = []
        sens_n_detach = []
        
        if input_size == 32: 
            # feature_steps = np.arange(1, 32)**2
            feature_steps = [1, 2, 4, 8, 16, 24, 28, 30, 31]

            # feature_steps = np.concatenate([np.arange(1, 10)**2, np.arange(100, 1001, 100)])
        elif input_size == 224:
            feature_steps = [1, 2, 4, 8, 16, 32, 64, 128]

        elif input_size == 96:
            feature_steps = [1, 2, 4, 8, 16, 32, 48, 64]

        else:
            assert(input_size == 64)
            # feature_steps = np.concatenate([np.arange(1, 15)**2, np.arange(200, 4001, 200)])
            feature_steps = [1, 2, 4, 8, 16, 32, 48, 56, 60, 62, 63]

        repeats = args["repeats"]
        for n_features in feature_steps:
            # get_n_sensitvity_blockified
            s, s_detach = get_n_sensitvity_blockified(model, batch, n_features, expl_postprocessing, grad_steps = grad_steps, repeats = repeats, baseline = baseline)

            # sens_n += [get_n_sensitivity_blockified(model, batch, n_features, expl_postprocessing, grad_steps = grad_steps, detach_ctx = False, repeats = repeats, baseline = baseline)]
            # sens_n_detach += [get_n_sensitivity(model, batch, n_features, expl_postprocessing, grad_steps = grad_steps, detach_ctx = True, repeats = repeats, baseline = baseline)]
            sens_n += [s]
            sens_n_detach += [s_detach]

        df_sens_n = pd.DataFrame.from_dict({"steps" : feature_steps, "with_ctx" : sens_n, "detached_ctx" : sens_n_detach}, orient = "index")
        
        print(df_sens_n)
        sens_n_path= f"{dataset_path}/sens_n_p{p}_gs{grad_steps}_{args['model']}"
        
        if clamp_negative:
            sens_n_path += "_cn"

        if posthoc:
            sens_n_path += "_posthoc"

        df_sens_n.to_pickle(sens_n_path)

if __name__ == "__main__":
    DEVICE = torch.device("mps" if sys.platform == "darwin" else "cuda")
    main()

