import time
import torch
import torchvision
from tqdm import tqdm


def timer(start, end):
    # Return the difference between <start> time and <end> time as a string in the format of "mm:ss"
    minutes, seconds = divmod(end - start, 60)
    return "{:0>2}:{:0>2}".format(int(minutes), int(seconds))


def train_step(dl, disc, gen, opt_gen, opt_disc, bce, vgg_loss, mse, device):
    total_disc_loss, total_gen_loss = 0, 0
    disc.train()
    gen.train()

    start = time.time()
    for idx, batch in enumerate(tqdm(dl, leave=True)):
        # Send data to GPU
        gt = batch["gt"].to(device)
        lr = batch["lq"].to(device)

        # Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
        # During discriminator model training, enable discriminator model backpropagation
        for d_parameters in disc.parameters():
            d_parameters.requires_grad = True

        # Calculate the classification score of the discriminator model for real samples
        disc_real = disc(gt)
        disc_loss_real = bce(disc_real, torch.ones_like(disc_real, device=device))

        # Calculate the classification score of the discriminator model for sr samples
        sr = gen(lr)
        disc_sr = disc(sr.detach())
        disc_loss_fake = bce(disc_sr, torch.zeros_like(disc_sr, device=device))

        # Discriminator loss = loss on pred high resolution images + loss on real high resolution images
        disc_loss = disc_loss_fake + disc_loss_real
        total_disc_loss += disc_loss  # save for printing

        # zero the previous gradients, find the new gradients, and apply gradient descent
        opt_disc.zero_grad()
        disc_loss.backward()
        opt_disc.step()

        # Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
        # During generator training, turn off discriminator backpropagation
        for d_parameters in disc.parameters():
            d_parameters.requires_grad = False

        # Generator loss has 3 parts: Adversarial loss + VGG loss + MSE loss
        disc_sr = disc(sr)
        adversarial_loss = 1e-3 * bce(disc_sr, torch.ones_like(disc_sr, device=device))
        loss_for_vgg = vgg_loss(sr, gt)
        pixel_loss = mse(sr, gt)
        gen_loss = loss_for_vgg + adversarial_loss + pixel_loss
        total_gen_loss += gen_loss  # save for printing

        # zero the previous gradients, find the new gradients, and apply gradient descent
        opt_gen.zero_grad()
        gen_loss.backward()
        opt_gen.step()

    end = time.time()
    # Calculates loss per epoch and print out what's happening
    total_disc_loss /= len(dl)
    total_gen_loss /= len(dl)
    print(f"Disc loss: {total_disc_loss:.5f} | Gen loss: {total_gen_loss:.2f}% | Time elapsed: {timer(start, end)}")


def test_step(gen, test_dl, device, writer, writer_step):
    gen.eval()  # put the generator in eval mode
    # turn on inference context manager
    with torch.inference_mode():
        # get a batch
        test_batch = next(iter(test_dl))

        # send data to GPU
        gt = test_batch["gt"].to(device)
        lr = test_batch["lq"].to(device)

        # run the generator
        sr = gen(lr)

        # save in tensorboard
        interleaved = torch.stack((gt, sr), dim=1).view(-1, *gt.shape[1:])
        img_grid = torchvision.utils.make_grid(interleaved)
        writer.add_image("GT High Resolution VS. PRED High Resolution", img_grid, step=writer_step)
