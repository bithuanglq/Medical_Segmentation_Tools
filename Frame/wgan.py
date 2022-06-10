
optimizer_G, optimizer_D, cuda

for epoch in range(opt.n_epochs):
    for i, (imgs, _) in enumerate(dataloader):

        real_imgs = Variable(imgs.type(Tensor))  # Configure input
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))   # Sample noise as generator input

        # ---------------------
        #  Train Discriminator
        # ---------------------

        fake_imgs = generator(z).detach()       # detach() 生成器不反向传播
        loss_D = -torch.mean(discriminator(real_imgs)) + torch.mean(discriminator(fake_imgs))

        ...

        
        for p in discriminator.parameters():                # Clip weights of discriminator  裁剪分辨器D的权重是WGAN的特点，加速收敛
            p.data.clamp_(-opt.clip_value, opt.clip_value)


        
        if epoch % 5 == 0:
            # -----------------
            #  Train Generator
            # -----------------

            gen_imgs = generator(z)      # Generate a batch of images
            loss_G = -torch.mean(discriminator(gen_imgs))           # Adversarial loss

            ...
