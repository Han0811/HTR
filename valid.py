import torch
import torch.nn.functional as F
import editdistance
from utils import utils as uutils

def validation(model, criterion, eval_loader, converter, device, args):
    model.eval()
    tot_loss = 0.0
    tot_cer = 0.0
    tot_wer = 0.0
    count = 0
    all_preds_str = []
    all_labels = []

    with torch.no_grad():
        for i, (images, labels) in enumerate(eval_loader):
            images = images.to(device)
            batch_size = images.size(0)
            boxes, score, logits = model(images)

            # Score loss
            if score is not None:
                target_score = torch.ones_like(score)
                score_loss = F.binary_cross_entropy(score, target_score)
            else:
                score_loss = torch.tensor(0.0, device=device)

            # Recognition loss
            if logits is not None:
                text, length = converter.encode(labels)
                preds = logits.float()
                preds_size = torch.IntTensor([preds.size(1)] * preds.size(0)).to(device)
                preds = preds.permute(1, 0, 2).log_softmax(2)
                torch.backends.cudnn.enabled = False
                rec_loss = criterion(preds, text.cuda(), preds_size, length.cuda()).mean()
                torch.backends.cudnn.enabled = True

                # decode
                _, preds_index = preds.max(2)
                preds_index = preds_index.transpose(1, 0).contiguous().view(-1)
                preds_str = converter.decode(preds_index.data, preds_size.data)
            else:
                rec_loss = torch.tensor(0.0, device=device)
                preds_str = [""]*len(labels)

            # YOLO loss placeholder
            yolo_loss = torch.tensor(0.0, device=device)

            loss = getattr(args, 'lambda_yolo', 0.0) * yolo_loss + getattr(args, 'lambda_score', 1.0) * score_loss + getattr(args, 'lambda_rec', 1.0) * rec_loss
            tot_loss += loss.item()

            # CER / WER
            for pred, gt in zip(preds_str, labels):
                tot_cer += editdistance.eval(pred, gt) / max(1, len(gt))
                pred_words = uutils.format_string_for_wer(pred).split()
                gt_words = uutils.format_string_for_wer(gt).split()
                tot_wer += editdistance.eval(pred_words, gt_words) / max(1, len(gt_words))

            all_preds_str.extend(preds_str)
            all_labels.extend(labels)
            count += 1

    avg_loss = tot_loss / count
    CER = tot_cer / count
    WER = tot_wer / count
    model.train()
    return avg_loss, CER, WER, all_preds_str, all_labels


# -------------------- train --------------------