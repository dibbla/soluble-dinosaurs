# a simple training script for a diffusion model

import os
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
import torchvision
from dotenv import load_dotenv
from model.unet2d import SimpleUNet2D
from train_scheme.ot_flow_matching import OTFlowMatching
from utils.exp_control import ExperimentController
from argparse import ArgumentParser

def main(args):
    # construct experiment controller with config
    config = vars(args)  # Convert args to dict for logging
    exp_controller = ExperimentController(
        experiment_name=f"{args.dataset}_{args.model}".replace("/", "_"),
        config=config,
    )

    # set device
    device = torch.device(args.device)
    exp_controller.logger.info(f"Using device: {device}")

    # load dataset
    exp_controller.logger.info(f"Loading dataset: {args.dataset}")
    dataset = load_dataset(
        args.dataset, 
        token=os.getenv("HUGGINGFACE_TOKEN"),
        cache_dir="datasets",
        split="train",
    )
    dataset.set_format(type="torch", columns=["image"])
    def transform(example):
        example["image"] = torchvision.transforms.Resize((args.train_size, args.train_size))(example["image"])
        return example
    dataset = dataset.map(transform, batched=True)
    train_dataset = dataset["image"]
    exp_controller.logger.info(f"Dataset loaded with {len(train_dataset)} images")

    # load model and optimizer
    model = SimpleUNet2D(num_blocks=3, in_channels=3, out_channels=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Log model architecture to TensorBoard
    exp_controller.log_model_graph(model, input_shape=(1, 3, args.train_size, args.train_size))

    # load dataloader
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.bs,
        shuffle=True,
        num_workers=4,
    )

    # train the model
    ot_flow_matching = OTFlowMatching()
    global_step = 0
    best_loss = float('inf')

    for epoch in range(args.epoch):
        model.train()
        epoch_losses = []
        
        for idx, batch in enumerate(train_loader):
            batch = batch.to(device) / 255.0  # normalize to [0, 1]
            noise_sample = torch.randn_like(batch)
            data_sample = batch
            
            # Forward pass
            loss = ot_flow_matching.loss(model, noise_sample, data_sample)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Log metrics
            loss_value = loss.item()
            epoch_losses.append(loss_value)
            exp_controller.log_metrics({'loss': loss_value}, global_step, phase='train')
            
            if idx % 100 == 0:
                exp_controller.logger.info(f"Epoch {epoch}, Step {idx}, Loss: {loss_value}")
            
            global_step += 1

        # Log epoch-level metrics
        epoch_avg_loss = sum(epoch_losses) / len(epoch_losses)
        exp_controller.log_metrics({'epoch_avg_loss': epoch_avg_loss}, epoch, phase='train')
        
        # Update best model if needed
        is_best = exp_controller.update_best_model(model, optimizer, epoch_avg_loss, epoch)
        
        # Save regular checkpoint
        if epoch % args.save_every == 0 or epoch == args.epoch - 1:
            exp_controller.save_model(model, optimizer, step=epoch)

        # Sample some generations
        model.eval()
        with torch.no_grad():
            for i in range(5):
                noise = torch.randn(1, batch.shape[1], args.train_size, args.train_size).to(device)
                generated_image = ot_flow_matching.generate(model, steps=25, noise=noise)
                # Save with experiment controller
                image_filename = f"generated_epoch{epoch}_sample{i}.png"
                exp_controller.save_image(
                    generated_image, 
                    image_filename, 
                    normalize=True,
                    tag=f"generation/epoch_{epoch}"
                )

    # Finalize experiment
    exp_controller.plot_metrics()
    exp_controller.finish()
    exp_controller.logger.info("Training completed!")

if __name__ == "__main__":
    load_dotenv()
    parser = ArgumentParser(description="Train a diffusion model")
    parser.add_argument("--dataset", type=str, default="korexyz/celeba-hq-256x256", help="Dataset to use in HF format")
    parser.add_argument("--model", type=str, default="unet2d", help="Model architecture")
    parser.add_argument("--device", type=str, default="cuda:7", help="Device to use for training")
    parser.add_argument("--bs", type=int, default=32, help="Batch size")
    parser.add_argument("--epoch", type=int, default=10, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--train_size", type=int, default=48, help="Size of training images")
    parser.add_argument("--save_every", type=int, default=2, help="Save checkpoints every N epochs")
    args = parser.parse_args()

    main(args)