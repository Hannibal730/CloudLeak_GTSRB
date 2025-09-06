import argparse, os, torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models.vgg19_deepid import VGG19DeepID
def main(args):
    tfm=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    test_ds=datasets.ImageFolder(os.path.join(args.data_dir,"test"), transform=tfm)
    test_loader=DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=2)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=VGG19DeepID(num_classes=len(test_ds.classes)).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device, weights_only=True))
    model.eval(); tot=cor=0
    with torch.no_grad():
        for x,y in test_loader:
            x,y=x.to(device), y.to(device); pred=model(x).argmax(1); cor+=(pred==y).sum().item(); tot+=y.numel()
    print(f"ACC={cor/max(1,tot):.4f}")
if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--data_dir",required=True); ap.add_argument("--ckpt",required=True)
    main(ap.parse_args())
