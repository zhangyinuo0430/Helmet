import sys
import torch
from ultralytics import YOLO

def parse_value(value):
    if value.lower() == 'true':
        return True
    elif value.lower() == 'false':
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value

def main():
    if len(sys.argv) < 3:
        print("Usage: python train_runner.py <model_path> <data_path> [--device <device>] [--epochs <epochs>] [--batch <batch>] [--imgsz <imgsz>] [--project <project>] [--name <name>]")
        sys.exit(1)
    
    model_path = sys.argv[1]
    data_path = sys.argv[2]
    
    device = 0 if torch.cuda.is_available() else 'cpu'
    epochs = 100
    batch = 4
    imgsz = 640
    project = 'runs/helmet_ablation'
    name = 'experiment'
    extra_args = {}
    
    i = 3
    while i < len(sys.argv):
        if sys.argv[i] == '--device':
            device = sys.argv[i+1]
            if device == '0' and not torch.cuda.is_available():
                device = 'cpu'
            i += 2
        elif sys.argv[i] == '--epochs':
            epochs = int(sys.argv[i+1])
            i += 2
        elif sys.argv[i] == '--batch':
            batch = int(sys.argv[i+1])
            i += 2
        elif sys.argv[i] == '--imgsz':
            imgsz = int(sys.argv[i+1])
            i += 2
        elif sys.argv[i] == '--project':
            project = sys.argv[i+1]
            i += 2
        elif sys.argv[i] == '--name':
            name = sys.argv[i+1]
            i += 2
        elif '=' in sys.argv[i]:
            key, value = sys.argv[i].split('=', 1)
            extra_args[key] = parse_value(value)
            i += 1
        else:
            i += 1
    
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        if isinstance(device, str) and device == '0':
            device = 0
    else:
        print("CUDA not available, using CPU")
        device = 'cpu'
    
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)
    
    train_args = {
        'data': data_path,
        'epochs': epochs,
        'imgsz': imgsz,
        'batch': batch,
        'device': device,
        'project': project,
        'name': name,
        'exist_ok': True,
        'seed': 42,
        'deterministic': True,
        'pretrained': False,
        'cache': False,
        'amp': True,
        'close_mosaic': 10,
        'mosaic': 1.0,
        'mixup': 0.10,
        'hsv_h': 0.015,
        'hsv_s': 0.70,
        'hsv_v': 0.40,
        'degrees': 5.0,
        'translate': 0.10,
        'scale': 0.75,
        'fliplr': 0.50,
        'plots': True,
        'val': True
    }
    
    train_args.update(extra_args)
    
    print(f"Starting training with device: {device}")
    results = model.train(**train_args)
    
    print(f"Training completed! Results saved to: {results.save_dir}")

if __name__ == '__main__':
    main()