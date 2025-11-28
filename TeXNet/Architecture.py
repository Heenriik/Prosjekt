import torch
from model import SMPModel
from config import parse_args
import sys
import os

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)

# Override sys.argv to provide default arguments
sys.argv = [
    'lol.py',
    '--data_dir', parent_dir,  # Use computed parent directory
    '--checkpoint_dir', 'temp',
    '--train_T',
    '--train_v',
    '--fold', '0'
]

# Use the actual parser
args = parse_args()

# Debug: print the final path
print(f"Data dir: {args.data_dir}")
print(f"Looking for: {os.path.join(args.data_dir, 'HADAR database')}")
print(f"Exists: {os.path.exists(os.path.join(args.data_dir, 'HADAR database'))}")

# Create model
model = SMPModel(args)

# Export to ONNX for Netron visualization
print("\nExporting model to ONNX format...")
dummy_input = torch.randn(1, 49, 256, 256)
torch.onnx.export(
    model.texnet, 
    dummy_input, 
    "texnet.onnx",
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})
print("Model exported to texnet.onnx - open with Netron to visualize")

# Print the entire architecture
print("="*50)
print("TeXNet Architecture:")
print("="*50)
print(model.texnet)

print("\n" + "="*50)
print("Model Summary:")
print("="*50)
print(model)

# Optional: count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nTotal parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")