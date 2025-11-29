# train_unified.py
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from functools import partial

from valid import validation
from utils import utils
from utils import sam
from utils import option
from data import dataset
from model.model import UnifiedHTR  

def compute_batch_loss(args, model, batch, converter, criterion, device):
    images = batch[0].to(device)
    labels = batch[1]
    batch_size = images.size(0)

    # forward
    boxes, score, logits = model(images)

    # YOLO loss: Ultralytics wrapper có tính loss nội bộ, nhưng ở đây chúng ta không truy xuất labels line-level cho YOLO, nên tạm 0
    yolo_loss = torch.tensor(0.0, device=device)

    # Score loss: nếu score != None
    if score is not None:
        # tạo target: assume tất cả line có gt = 1
        target_score = torch.ones_like(score)
        score_loss = F.binary_cross_entropy(score, target_score)
    else:
        score_loss = torch.tensor(0.0, device=device)

    # Recognition loss
    if logits is not None:
        text, length = converter.encode([l for l in labels])
        preds = logits.float()
        preds_size = torch.IntTensor([preds.size(1)] * preds.size(0)).to(device)
        preds = preds.permute(1, 0, 2).log_softmax(2)

        torch.backends.cudnn.enabled = False
        rec_loss = criterion(preds, text.cuda(), preds_size, length.cuda()).mean()
        torch.backends.cudnn.enabled = True
    else:
        rec_loss = torch.tensor(0.0, device=device)

    total_loss = args.lambda_yolo * yolo_loss + args.lambda_score * score_loss + args.lambda_rec * rec_loss

    return total_loss, yolo_loss, score_loss, rec_loss


def main():
    args = option.get_args_parser()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(args.seed)

    # save dir
    args.save_dir = os.path.join(args.out_dir, args.exp_name)
    os.makedirs(args.save_dir, exist_ok=True)
    writer = SummaryWriter(args.save_dir)
    logger = utils.get_logger(args.save_dir)
    logger.info(json.dumps(vars(args), indent=4, sort_keys=True))

    # Model
    model = UnifiedHTR(yolo_weights=args.yolo_weights,
                       num_classes=args.nb_cls,
                       img_size=args.img_size).to(device)

    # EMA
    model_ema = utils.ModelEma(model, args.ema_decay)

    # Data
    train_dataset = dataset.myLoadDS(args.train_data_list, args.data_path, args.img_size)
    train_loader = DataLoader(train_dataset,
                              batch_size=args.train_bs,
                              shuffle=True,
                              pin_memory=True,
                              num_workers=args.num_workers,
                              collate_fn=partial(dataset.SameTrCollate, args=args))
    train_iter = dataset.cycle_data(train_loader)

    val_dataset = dataset.myLoadDS(args.val_data_list, args.data_path, args.img_size, ralph=train_dataset.ralph)
    val_loader = DataLoader(val_dataset,
                            batch_size=args.val_bs,
                            shuffle=False,
                            pin_memory=True,
                            num_workers=args.num_workers)

    # Optimizer + SAM
    optimizer = sam.SAM(model.parameters(), torch.optim.AdamW, lr=args.max_lr, betas=(0.9,0.99), weight_decay=args.weight_decay)
    criterion = nn.CTCLoss(reduction='none', zero_infinity=True)
    converter = utils.CTCLabelConverter(train_dataset.ralph.values())

    best_cer, best_wer = 1e9, 1e9
    train_loss = 0.0

    for nb_iter in range(1, args.total_iter+1):
        optimizer, current_lr = utils.update_lr_cos(nb_iter, args.warm_up_iter, args.total_iter, args.max_lr, optimizer)

        batch = next(train_iter)
        loss, yolo_loss, score_loss, rec_loss = compute_batch_loss(args, model, batch, converter, criterion, device)

        loss.backward()
        optimizer.first_step(zero_grad=True)
        # second step
        loss2, _, _, _ = compute_batch_loss(args, model, batch, converter, criterion, device)
        loss2.backward()
        optimizer.second_step(zero_grad=True)
        model.zero_grad()
        model_ema.update(model, num_updates=nb_iter)

        train_loss += loss.item()
        if nb_iter % args.print_iter == 0:
            avg_loss = train_loss / args.print_iter
            logger.info(f"Iter {nb_iter} LR {current_lr:.6f} total_loss {avg_loss:.6f} yolo_loss {yolo_loss.item():.6f} score_loss {score_loss.item():.6f} rec_loss {rec_loss.item():.6f}")
            writer.add_scalar('Train/total_loss', avg_loss, nb_iter)
            writer.add_scalar('Train/yolo_loss', yolo_loss.item(), nb_iter)
            writer.add_scalar('Train/score_loss', score_loss.item(), nb_iter)
            writer.add_scalar('Train/rec_loss', rec_loss.item(), nb_iter)
            train_loss = 0.0

        if nb_iter % args.eval_iter == 0:
            val_loss, CER, WER, _, _ = validation(model_ema.ema, criterion, val_loader, converter, device, args)
            logger.info(f"[VAL] loss:{val_loss:.4f} CER:{CER:.4f} WER:{WER:.4f}")

            writer.add_scalar('VAL/loss', val_loss, nb_iter)
            writer.add_scalar('VAL/CER', CER, nb_iter)
            writer.add_scalar('VAL/WER', WER, nb_iter)

            # checkpoint
            if CER < best_cer:
                best_cer = CER
                torch.save({'model': model.state_dict(), 'ema': model_ema.ema.state_dict(), 'optimizer': optimizer.state_dict()}, os.path.join(args.save_dir, 'best_CER.pth'))

            if WER < best_wer:
                best_wer = WER
                torch.save({'model': model.state_dict(), 'ema': model_ema.ema.state_dict(), 'optimizer': optimizer.state_dict()}, os.path.join(args.save_dir, 'best_WER.pth'))

            # periodic
            torch.save({'model': model.state_dict(), 'iter': nb_iter}, os.path.join(args.save_dir, f'checkpoint_{nb_iter}.pth'))


if __name__ == "__main__":
    main()
