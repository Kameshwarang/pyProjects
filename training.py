import torch
import numpy as np
from matplotlib import pyplot as plt

from ODEL1 import SiameseNetStereoMatching, similarity_score
from loss import hinge_loss
from stereo_batch_provider import KITTIDisparityDataset, PatchProvider
import os
import datetime


def training(model, simialrity_score_calcualtor, batch_provider_train, batch_size, max_iterations, optimizer, device, batch_provider_val=None, eval_period=200):

    print('training started...')

    # move model to device
    model.to(device=device)

    # put model in training mode
    model.train()

    losses = []
    train_accuracies = []
    eval_losses = []
    eval_acc = []

    # feed a batch to the model and update weights
    for iter_i, batch_i in zip(range(max_iterations), batch_provider_train.iterate_batches(batch_size)):

        # get three batches: reference patches in the left image, their corresponding positive patches, their corresponding negative patches
        batch_ref_patches, batch_pos_patches, batch_neg_patches = batch_i

        # convert shapes from (batch, height, width, channel) to (batch, channel, height, width)
        batch_ref_patches = batch_ref_patches.permute(0, 3, 1, 2)
        batch_pos_patches = batch_pos_patches.permute(0, 3, 1, 2)
        batch_neg_patches = batch_neg_patches.permute(0, 3, 1, 2)

        # move to the device
        batch_ref_patches = batch_ref_patches.to(device)
        batch_pos_patches = batch_pos_patches.to(device)
        batch_neg_patches = batch_neg_patches.to(device)

        # calculate similarity of reference patches in the left image to positive and negative patches in the right image
        pos_sim = simialrity_score_calcualtor(model, x_l=batch_ref_patches, x_r=batch_pos_patches)
        neg_sim = simialrity_score_calcualtor(model, x_l=batch_ref_patches, x_r=batch_neg_patches)

        # calculate hinge loss
        batch_mean_loss, batch_accuracy = hinge_loss(pos_patch_similarity=pos_sim, neg_patch_similarity=neg_sim, margin=0.2)

        losses.append(batch_mean_loss.item())
        train_accuracies.append(batch_accuracy.item())

        # compute gradients
        optimizer.zero_grad()
        batch_mean_loss.backward()

        # update weights
        optimizer.step()

        if (iter_i % 100 == 0) or (iter_i == (max_iterations - 1)):
            print('=' * 30)
            print('Iteration {}'.format(iter_i))
            print('Train average loss in the last 100 iterations:           {}'.format(sum(losses[-100:]) / 100))
            print('Train average train accuracy in the last 100 iterations: {}'.format(
                sum(train_accuracies[-100:]) / 100))

        # evaluate on val data
        if (iter_i % eval_period == 0) or (iter_i == (max_iterations - 1)):
            if batch_provider_val is not None:
                model.eval()
                with torch.no_grad():
                    for iter_val_i, batch_val_i in zip(range(5), batch_provider_val.iterate_batches(batch_size)):
                        # get three batches: reference patches in the left image, their corresponding positive patches, their corresponding negative patches
                        batch_ref_patches, batch_pos_patches, batch_neg_patches = batch_val_i
                        # convert shapes from (batch, height, width, channel) to (batch, channel, height, width)
                        batch_ref_patches = batch_ref_patches.permute(0, 3, 1, 2)
                        batch_pos_patches = batch_pos_patches.permute(0, 3, 1, 2)
                        batch_neg_patches = batch_neg_patches.permute(0, 3, 1, 2)
                        # move to the device
                        batch_ref_patches = batch_ref_patches.to(device)
                        batch_pos_patches = batch_pos_patches.to(device)
                        batch_neg_patches = batch_neg_patches.to(device)
                        # calculate similarity of reference patches in the left image to positive and negative patches in the right image
                        pos_sim = simialrity_score_calcualtor(model, x_l=batch_ref_patches, x_r=batch_pos_patches)
                        neg_sim = simialrity_score_calcualtor(model, x_l=batch_ref_patches, x_r=batch_neg_patches)
                        # calculate hinge loss
                        batch_mean_loss_eval, batch_accuracy_eval = hinge_loss(pos_patch_similarity=pos_sim,
                                                                              neg_patch_similarity=neg_sim, margin=0.2)
                        eval_losses.append(batch_mean_loss_eval.item())
                        eval_acc.append(batch_accuracy_eval.item())
                print('Val average loss                       :             {}'.format(sum(eval_losses) / len(eval_losses)))
                print('Val accuracy                           :             {}'.format(sum(eval_acc) / len(eval_acc)))
                model.train()

    # put model in evaluation mode
    model.eval()

    print('training finished.')

    return losses, train_accuracies, eval_losses, eval_acc


if _name_ == '_main_':

    np.random.seed(0)
    torch.manual_seed(0)

    # hyperparameters for training Siamese patch matcher
    hyper_param = {'path_size': (9, 9), 'max_iteration': 1500, 'batch_size': 128, 'learning_rate': 2e-4}

    # output directory
    current_datetime = datetime.datetime.now()
    formatted_date = current_datetime.strftime('%Y-%m-%d')
    formatted_time = current_datetime.strftime('%H-%M-%S')
    output_directory = './saved-model-' + formatted_date + '-' + formatted_time
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Device: {}'.format(device))

    # create model
    model = SiameseNetStereoMatching(in_channel=1)

    # KITTI train disparity dataset
    dataset_train = KITTIDisparityDataset(
        image_dir='./KITTI_disparity_subset/data_scene_flow/training',
        disparity_dir='./KITTI_disparity_subset/data_scene_flow/training/disp_noc_0/',
        downsample=True)
    # batch patch provider
    patch_provider_obj_train = PatchProvider(data=dataset_train, patch_size=hyper_param['path_size'], N=(4, 10), P=1)

    # KITTI val disparity dataset
    dataset_val = KITTIDisparityDataset(
        image_dir='./KITTI_disparity_subset/data_scene_flow/val',
        disparity_dir='./KITTI_disparity_subset/data_scene_flow/val/disp_noc_0/',
        downsample=True)
    # batch patch provider
    patch_provider_obj_val = PatchProvider(data=dataset_val, patch_size=hyper_param['path_size'], N=(4, 10), P=1)

    # Siamese model
    model = SiameseNetStereoMatching(in_channel=1)

    # optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=hyper_param['learning_rate'], momentum=0.9)

    # train model
    losses, train_accuracies, eval_losses, eval_acc = training(model=model,
                                                               simialrity_score_calcualtor=similarity_score,
                                                               batch_provider_train=patch_provider_obj_train,
                                                               batch_size=hyper_param['batch_size'],
                                                               max_iterations=hyper_param['max_iteration'],
                                                               optimizer=optimizer,
                                                               device=device,
                                                               batch_provider_val=patch_provider_obj_val)
    # Plotting
    plt.figure(figsize=(12, 6))

    # Training loss plot
    plt.subplot(1, 2, 1)
    plt.plot(losses, label='Training Loss')
    plt.title('Training Loss over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Training accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.title('Training Accuracy over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Validation loss and accuracy plot
    plt.figure(figsize=(12, 6))

    # Validation loss plot
    plt.subplot(1, 2, 1)
    plt.plot(eval_losses, label='Validation Loss')
    plt.title('Validation Loss over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()

    # Validation accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(eval_acc, label='Validation Accuracy')
    plt.title('Validation Accuracy over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # stop threads
    patch_provider_obj_train.stop()

    # Specify the directory path where you want to save the model
    model_save_path = os.path.join(output_directory, 'model.pth')

    # Save the model to the specified path
    torch.save(model.state_dict(), model_save_path)
    print('Model was saved.')