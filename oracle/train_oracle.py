import os, argparse, torch, torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from oracle.resnet import get_resnet50
IMAGENET_MEAN=[0.485,0.456,0.406]; IMAGENET_STD=[0.229,0.224,0.225]
def get_loader(data_dir, split="train", batch_size=64, size=224, workers=4):
    tfm=transforms.Compose([transforms.Resize((size,size)),transforms.ToTensor(),transforms.Normalize(IMAGENET_MEAN,IMAGENET_STD)])
    ds=datasets.ImageFolder(os.path.join(data_dir,split),transform=tfm)
    return DataLoader(ds,batch_size=batch_size,shuffle=(split=="train"),num_workers=workers), ds
def main(args):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader,train_ds=get_loader(args.data_dir,"train",args.batch_size,args.size,args.workers)
    test_loader, test_ds =get_loader(args.data_dir,"test", args.batch_size,args.size,args.workers)
    model=get_resnet50(num_classes=len(train_ds.classes)).to(device)
    optim=torch.optim.SGD(model.parameters(),lr=args.lr,momentum=0.9,weight_decay=1e-4)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(optim,T_max=args.epochs)
    crit=nn.CrossEntropyLoss(); best=0.0
    for ep in range(1,args.epochs+1):
        model.train(); tot=cor=0; loss_sum=0.0
        for x,y in train_loader:
            x,y=x.to(device),y.to(device); optim.zero_grad(set_to_none=True)
            logits=model(x); loss=crit(logits,y); loss.backward(); optim.step()
            loss_sum+=loss.item()*y.size(0); cor+=(logits.argmax(1)==y).sum().item(); tot+=y.numel()
        sched.step(); model.eval(); ttot=tcor=0
        with torch.no_grad():
            for x,y in test_loader:
                x,y=x.to(device),y.to(device); ll=model(x); tcor+=(ll.argmax(1)==y).sum().item(); ttot+=y.numel()
        acc=tcor/max(1,ttot); print(f"[{ep}/{args.epochs}] loss={loss_sum/max(1,len(train_ds)):.4f} train_acc={cor/max(1,tot):.3f} test_acc={acc:.3f}")
        if acc>best: best=acc; os.makedirs(os.path.dirname(args.out),exist_ok=True); torch.save(model.state_dict(),args.out); print("  -> saved:",args.out)
if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--data_dir",required=True); ap.add_argument("--out",default="models_ckpt/resnet50_gtsrb.pth")
    ap.add_argument("--epochs",type=int,default=10); ap.add_argument("--batch_size",type=int,default=64)
    ap.add_argument("--size",type=int,default=224); ap.add_argument("--workers",type=int,default=4)
    ap.add_argument("--lr",type=float,default=0.01); args=ap.parse_args(); main(args)
