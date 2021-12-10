import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import glob
import json
import tqdm
import functools

from torch.utils.data.distributed import DistributedSampler
from einops import rearrange

from vilt.modules.dist_utils import all_gather


def cost_matrix_cosine(x, y, eps=1e-5):
    """Compute cosine distnace across every pairs of x, y (batched)
    [B, L_x, D] [B, L_y, D] -> [B, Lx, Ly]"""
    assert x.dim() == y.dim()
    assert x.size(0) == y.size(0)
    assert x.size(2) == y.size(2)
    x_norm = F.normalize(x, p=2, dim=-1, eps=eps)
    y_norm = F.normalize(y, p=2, dim=-1, eps=eps)
    cosine_sim = x_norm.matmul(y_norm.transpose(1, 2))
    cosine_dist = 1 - cosine_sim
    return cosine_dist


def trace(x):
    """ compute trace of input tensor (batched) """
    b, m, n = x.size()
    assert m == n
    mask = torch.eye(n, dtype=torch.bool, device=x.device).unsqueeze(0).expand_as(x)
    trace = x.masked_select(mask).contiguous().view(b, n).sum(dim=-1, keepdim=False)
    return trace


@torch.no_grad()
def ipot(C, x_len, x_pad, y_len, y_pad, joint_pad, beta, iteration, k):
    """ [B, M, N], [B], [B, M], [B], [B, N], [B, M, N]"""
    b, m, n = C.size()
    sigma = torch.ones(b, m, dtype=C.dtype, device=C.device) / x_len.unsqueeze(1)
    T = torch.ones(b, n, m, dtype=C.dtype, device=C.device)
    A = torch.exp(-C.transpose(1, 2) / beta)

    # mask padded positions
    sigma.masked_fill_(x_pad, 0)
    joint_pad = joint_pad.transpose(1, 2)
    T.masked_fill_(joint_pad, 0)
    A.masked_fill_(joint_pad, 0)

    # broadcastable lengths
    x_len = x_len.unsqueeze(1).unsqueeze(2)
    y_len = y_len.unsqueeze(1).unsqueeze(2)

    # mask to zero out padding in delta and sigma
    x_mask = (x_pad.to(C.dtype) * 1e4).unsqueeze(1)
    y_mask = (y_pad.to(C.dtype) * 1e4).unsqueeze(1)

    for _ in range(iteration):
        Q = A * T  # bs * n * m
        sigma = sigma.view(b, m, 1)
        for _ in range(k):
            delta = 1 / (y_len * Q.matmul(sigma).view(b, 1, n) + y_mask)
            sigma = 1 / (x_len * delta.matmul(Q) + x_mask)
        T = delta.view(b, n, 1) * Q * sigma
    T.masked_fill_(joint_pad, 0)
    return T


def optimal_transport_dist(
    txt_emb, img_emb, txt_pad, img_pad, beta=0.5, iteration=50, k=1
):
    """ [B, M, D], [B, N, D], [B, M], [B, N]"""
    cost = cost_matrix_cosine(txt_emb, img_emb)
    # mask the padded inputs
    joint_pad = txt_pad.unsqueeze(-1) | img_pad.unsqueeze(-2)
    cost.masked_fill_(joint_pad, 0)

    txt_len = (txt_pad.size(1) - txt_pad.sum(dim=1, keepdim=False)).to(dtype=cost.dtype)
    img_len = (img_pad.size(1) - img_pad.sum(dim=1, keepdim=False)).to(dtype=cost.dtype)

    T = ipot(
        cost.detach(), txt_len, txt_pad, img_len, img_pad, joint_pad, beta, iteration, k
    )
    distance = trace(cost.matmul(T.detach()))
    return distance


def compute_mlm(pl_module, batch):
    infer = pl_module.infer(batch, mask_text=True, mask_image=False)
    mlm_logits = pl_module.mlm_score(infer["text_feats"])
    mlm_labels = infer["text_labels"]

    mlm_loss = F.cross_entropy(
        mlm_logits.view(-1, pl_module.hparams.config["vocab_size"]),
        mlm_labels.view(-1),
        ignore_index=-100,
    )

    ret = {
        "mlm_loss": mlm_loss,
        "mlm_logits": mlm_logits,
        "mlm_labels": mlm_labels,
        "mlm_ids": infer["text_ids"],
    }

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_mlm_loss")(ret["mlm_loss"])
    acc = getattr(pl_module, f"{phase}_mlm_accuracy")(
        ret["mlm_logits"], ret["mlm_labels"]
    )

    return ret


def compute_mpp(pl_module, batch):
    infer = pl_module.infer(batch, mask_text=False, mask_image=True)
    mpp_logits = pl_module.mpp_score(infer["image_feats"])
    mpp_logits = torch.stack(
        [
            mpp_logits[:, :, 0:256],
            mpp_logits[:, :, 256:512],
            mpp_logits[:, :, 512:768],
        ],
        dim=2,
    )
    mpp_labels = infer["image_labels"]

    mpp_loss = F.cross_entropy(
        mpp_logits.view(-1, 256),
        mpp_labels.view(-1),
        ignore_index=-100,
    )

    ret = {
        "mpp_loss": mpp_loss,
        "mpp_logits": mpp_logits,
        "mpp_labels": mpp_labels,
    }

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_mpp_loss")(ret["mpp_loss"])
    acc = getattr(pl_module, f"{phase}_mpp_accuracy")(
        ret["mpp_logits"], ret["mpp_labels"]
    )
    pl_module.log(f"mpp/{phase}/loss", loss)
    pl_module.log(f"mpp/{phase}/accuracy", acc)

    return ret


def compute_mppd(pl_module, batch):
    infer = pl_module.infer(batch, mask_text=False, mask_image=True)
    mppd_logits = pl_module.mppd_score(infer["image_feats"])
    mppd_labels = infer["image_labels_mppd"]
    filter_to_train = infer["image_labels"].float().mean(dim=-1) != -100

    labels = mppd_labels[filter_to_train]
    logits = mppd_logits[filter_to_train]
    mppd_loss = F.mse_loss(logits, labels)

    ret = {
        "mppd_loss": mppd_loss,
        "mppd_logits": mppd_logits,
        "mppd_labels": mppd_labels,
    }

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_mppd_loss")(ret["mppd_loss"])
    pl_module.log(f"mppd/{phase}/loss", loss)

    return ret


def compute_mpfr(pl_module, batch):
    infer = pl_module.infer(batch, mask_text=False, mask_image=True)
    mpfr_logits = pl_module.mpfr_score(infer["image_feats"])
    mpfr_labels = infer["image_labels_mpfr"]
    filter_to_train = infer["image_labels"].float().mean(dim=-1) != -100

    labels = mpfr_labels[filter_to_train]
    logits = mpfr_logits[filter_to_train]
    mpfr_loss = F.mse_loss(logits, labels)

    ret = {
        "mpfr_loss": mpfr_loss,
        "mpfr_logits": mpfr_logits,
        "mpfr_labels": mpfr_labels,
    }

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_mpfr_loss")(ret["mpfr_loss"])
    pl_module.log(f"mpfr/{phase}/loss", loss)

    return ret


def compute_itm_wpa(pl_module, batch):
    pos_len = len(batch["text"]) // 2
    neg_len = len(batch["text"]) - pos_len
    itm_labels = torch.cat([torch.ones(pos_len), torch.zeros(neg_len)]).to(
        get_current_device()
    )
    itm_labels = itm_labels[torch.randperm(itm_labels.size(0))]

    itm_images = [
        torch.stack(
            [
                ti if itm_labels[i] == 1 else fi
                for i, (ti, fi) in enumerate(zip(bti, bfi))
            ]
        )
        for bti, bfi in zip(batch["image"], batch["false_image_0"])
    ]

    batch = {k: v for k, v in batch.items()}
    batch["image"] = itm_images

    infer = pl_module.infer(batch, mask_text=False, mask_image=False)

    with torch.cuda.amp.autocast(enabled=False):
        txt_emb, img_emb = infer["text_feats"], infer["image_feats"]
        txt_mask, img_mask = infer["text_masks"].bool(), infer["image_masks"].bool()
        for i, _len in enumerate(txt_mask.sum(dim=1)):
            txt_mask[i, _len - 1] = False
        txt_mask[:, 0] = False
        img_mask[:, 0] = False
        if "deit" in pl_module.hparams.config["vit"]:
            img_mask[:, 1] = False
        txt_pad, img_pad = ~txt_mask, ~img_mask

        cost = cost_matrix_cosine(txt_emb.float(), img_emb.float())
        joint_pad = txt_pad.unsqueeze(-1) | img_pad.unsqueeze(-2)
        cost.masked_fill_(joint_pad, 0)

        txt_len = (txt_pad.size(1) - txt_pad.sum(dim=1, keepdim=False)).to(
            dtype=cost.dtype
        )
        img_len = (img_pad.size(1) - img_pad.sum(dim=1, keepdim=False)).to(
            dtype=cost.dtype
        )
        T = ipot(
            cost.detach(), txt_len, txt_pad, img_len, img_pad, joint_pad, 0.5, 50, 1
        )
        distance = trace(cost.matmul(T.detach()))

    dist_pos = distance.masked_select(itm_labels == 1)
    dist_neg = distance.masked_select(itm_labels == 0)
    ot_loss = (dist_pos.sum() - dist_neg.sum()) / (dist_pos.size(0) + dist_neg.size(0))

    itm_logits = pl_module.itm_score(infer["cls_feats"])
    itm_loss = F.cross_entropy(itm_logits, itm_labels.long())

    ret = {
        "itm_loss": itm_loss,
        "itm_wpa_loss": 0.1 * ot_loss,
        "itm_logits": itm_logits,
        "itm_labels": itm_labels,
    }

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_itm_loss")(ret["itm_loss"])
    wpa_loss = getattr(pl_module, f"{phase}_itm_wpa_loss")(ret["itm_wpa_loss"])
    acc = getattr(pl_module, f"{phase}_itm_accuracy")(
        ret["itm_logits"], ret["itm_labels"]
    )
    # pl_module.log(f"itm/{phase}/loss", loss)
    # pl_module.log(f"itm/{phase}/wpa_loss", wpa_loss)
    # pl_module.log(f"itm/{phase}/accuracy", acc)

    return ret


def compute_imgcls(pl_module, batch):
    infer = pl_module.infer(batch, mask_text=False, mask_image=False)
    imgcls_logits = pl_module.img_classifier(infer["cls_feats"])
    imgcls_labels = batch["label"]
    imgcls_labels = torch.tensor(imgcls_labels).to(get_current_device()).long()
    imgcls_loss = F.cross_entropy(imgcls_logits, imgcls_labels)

    ret = {
        "imgcls_loss": imgcls_loss,
        "imgcls_logits": imgcls_logits,
        "imgcls_labels": imgcls_labels,
    }

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_imgcls_loss")(ret["imgcls_loss"])
    acc = getattr(pl_module, f"{phase}_imgcls_accuracy")(
        ret["imgcls_logits"], ret["imgcls_labels"]
    )
    pl_module.log(f"imgcls/{phase}/loss", loss)
    pl_module.log(f"imgcls/{phase}/accuracy", acc)

    return ret


def compute_vqa(pl_module, batch):
    infer = pl_module.infer(batch, mask_text=False, mask_image=False)
    vqa_logits = pl_module.vqa_classifier(infer["cls_feats"])
    vqa_targets = torch.zeros(
        len(vqa_logits), pl_module.hparams.config["vqav2_label_size"]
    ).to(get_current_device())

    vqa_labels = batch["vqa_labels"]
    vqa_scores = batch["vqa_scores"]

    for i, (_label, _score) in enumerate(zip(vqa_labels, vqa_scores)):
        for l, s in zip(_label, _score):
            vqa_targets[i, l] = s

    vqa_loss = (
        F.binary_cross_entropy_with_logits(vqa_logits, vqa_targets)
        * vqa_targets.shape[1]
    )  # https://github.com/jnhwkim/ban-vqa/blob/master/train.py#L19

    ret = {
        "vqa_loss": vqa_loss,
        "vqa_logits": vqa_logits,
        "vqa_targets": vqa_targets,
        "vqa_labels": vqa_labels,
        "vqa_scores": vqa_scores,
    }

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_vqa_loss")(ret["vqa_loss"])
    score = getattr(pl_module, f"{phase}_vqa_score")(
        ret["vqa_logits"], ret["vqa_targets"]
    )
    pl_module.log(f"vqa/{phase}/loss", loss)
    pl_module.log(f"vqa/{phase}/score", score)

    return ret


def compute_nlvr2(pl_module, batch):
    infer1 = pl_module.infer(
        batch, mask_text=False, mask_image=False, image_token_type_idx=1
    )
    infer2 = pl_module.infer(
        batch, mask_text=False, mask_image=False, image_token_type_idx=2
    )

    cls_feats = torch.cat([infer1["cls_feats"], infer2["cls_feats"]], dim=-1)
    nlvr2_logits = pl_module.nlvr2_classifier(cls_feats)

    nlvr2_labels = batch["answers"]
    nlvr2_labels = torch.tensor(nlvr2_labels).to(get_current_device()).long()
    nlvr2_loss = F.cross_entropy(nlvr2_logits, nlvr2_labels)

    ret = {
        "nlvr2_loss": nlvr2_loss,
        "nlvr2_logits": nlvr2_logits,
        "nlvr2_labels": nlvr2_labels,
    }

    phase = "train" if pl_module.training else "val"

    if phase == "train":
        loss = getattr(pl_module, f"{phase}_nlvr2_loss")(ret["nlvr2_loss"])
        acc = getattr(pl_module, f"{phase}_nlvr2_accuracy")(
            ret["nlvr2_logits"], ret["nlvr2_labels"]
        )
        pl_module.log(f"nlvr2/{phase}/loss", loss)
        pl_module.log(f"nlvr2/{phase}/accuracy", acc)
    else:
        dev_batches = [i for i, n in enumerate(batch["table_name"]) if "dev" in n]
        test_batches = [i for i, n in enumerate(batch["table_name"]) if "test" in n]

        if dev_batches:
            dev_loss = getattr(pl_module, f"dev_nlvr2_loss")(
                F.cross_entropy(
                    ret["nlvr2_logits"][dev_batches], ret["nlvr2_labels"][dev_batches]
                )
            )
            dev_acc = getattr(pl_module, f"dev_nlvr2_accuracy")(
                ret["nlvr2_logits"][dev_batches], ret["nlvr2_labels"][dev_batches]
            )
            pl_module.log(f"nlvr2/dev/loss", dev_loss)
            pl_module.log(f"nlvr2/dev/accuracy", dev_acc)
        if test_batches:
            test_loss = getattr(pl_module, f"test_nlvr2_loss")(
                F.cross_entropy(
                    ret["nlvr2_logits"][test_batches], ret["nlvr2_labels"][test_batches]
                )
            )
            test_acc = getattr(pl_module, f"test_nlvr2_accuracy")(
                ret["nlvr2_logits"][test_batches], ret["nlvr2_labels"][test_batches]
            )
            pl_module.log(f"nlvr2/test/loss", test_loss)
            pl_module.log(f"nlvr2/test/accuracy", test_acc)

    return ret


def compute_irtr(pl_module, batch):
    is_training_phase = pl_module.training

    _bs, _c, _h, _w = batch["image"][0].shape
    false_len = pl_module.hparams.config["draw_false_text"]
    text_ids = torch.stack(
        [batch[f"false_text_{i}_ids"] for i in range(false_len)], dim=1
    )
    text_masks = torch.stack(
        [batch[f"false_text_{i}_masks"] for i in range(false_len)], dim=1
    )
    text_labels = torch.stack(
        [batch[f"false_text_{i}_labels"] for i in range(false_len)], dim=1
    )

    text_ids = torch.cat([batch["text_ids"].unsqueeze(1), text_ids], dim=1)
    text_masks = torch.cat([batch["text_masks"].unsqueeze(1), text_masks], dim=1)
    text_labels = torch.cat([batch["text_labels"].unsqueeze(1), text_labels], dim=1)
    images = batch["image"][0].unsqueeze(1).expand(_bs, false_len + 1, _c, _h, _w)

    infer = pl_module.infer(
        {
            "image": [rearrange(images, "bs fs c h w -> (bs fs) c h w")],
            "text_ids": rearrange(text_ids, "bs fs tl -> (bs fs) tl"),
            "text_masks": rearrange(text_masks, "bs fs tl -> (bs fs) tl"),
            "text_labels": rearrange(text_labels, "bs fs tl -> (bs fs) tl"),
        }
    )
    score = pl_module.rank_output(infer["cls_feats"])[:, 0]
    score = rearrange(score, "(bs fs) -> bs fs", bs=_bs, fs=false_len + 1)
    answer = torch.zeros(_bs).to(score).long()
    irtr_loss = F.cross_entropy(score, answer)

    ret = {
        "irtr_loss": irtr_loss,
    }

    phase = "train" if pl_module.training else "val"
    irtr_loss = getattr(pl_module, f"{phase}_irtr_loss")(ret["irtr_loss"])

    pl_module.log(f"irtr/{phase}/irtr_loss", irtr_loss)

    return ret


@torch.no_grad()
def compute_irtr_recall(pl_module):
    text_dset = pl_module.trainer.datamodule.dms[0].make_no_false_val_dset()
    text_dset.tokenizer = pl_module.trainer.datamodule.dms[0].tokenizer
    text_loader = torch.utils.data.DataLoader(
        text_dset,
        batch_size=64,
        num_workers=pl_module.hparams.config["num_workers"],
        pin_memory=True,
        collate_fn=functools.partial(
            text_dset.collate,
            mlm_collator=pl_module.trainer.datamodule.dms[0].mlm_collator,
        ),
    )

    image_dset = pl_module.trainer.datamodule.dms[0].make_no_false_val_dset(
        image_only=True
    )
    image_dset.tokenizer = pl_module.trainer.datamodule.dms[0].tokenizer
    dist_sampler = DistributedSampler(image_dset, shuffle=False)
    image_loader = torch.utils.data.DataLoader(
        image_dset,
        batch_size=1,
        num_workers=pl_module.hparams.config["num_workers"],
        sampler=dist_sampler,
        pin_memory=True,
        collate_fn=functools.partial(
            image_dset.collate,
            mlm_collator=pl_module.trainer.datamodule.dms[0].mlm_collator,
        ),
    )

    text_preload = list()
    for _b in tqdm.tqdm(text_loader, desc="text prefetch loop"):
        text_preload.append(
            {
                "text_ids": _b["text_ids"].to(get_current_device()),
                "text_masks": _b["text_masks"].to(get_current_device()),
                "text_labels": _b["text_labels"].to(get_current_device()),
                "img_index": _b["img_index"],
            }
        )

    tiids = list()
    for pre in text_preload:
        tiids += pre["img_index"]
    tiids = torch.tensor(tiids)

    image_preload = list()
    for _b in tqdm.tqdm(image_loader, desc="image prefetch loop"):
        (ie, im, _, _) = pl_module.transformer.visual_embed(
            _b["image"][0].to(get_current_device()),
            max_image_len=pl_module.hparams.config["max_image_len"],
            mask_it=False,
        )
        image_preload.append((ie, im, _b["img_index"][0]))

    rank_scores = list()
    rank_iids = list()

    for img_batch in tqdm.tqdm(image_preload, desc="rank loop"):
        _ie, _im, _iid = img_batch
        _, l, c = _ie.shape

        img_batch_score = list()
        for txt_batch in text_preload:
            fblen = len(txt_batch["text_ids"])
            ie = _ie.expand(fblen, l, c)
            im = _im.expand(fblen, l)

            with torch.cuda.amp.autocast():
                score = pl_module.rank_output(
                    pl_module.infer(
                        {
                            "text_ids": txt_batch["text_ids"],
                            "text_masks": txt_batch["text_masks"],
                            "text_labels": txt_batch["text_labels"],
                        },
                        image_embeds=ie,
                        image_masks=im,
                    )["cls_feats"]
                )[:, 0]

            img_batch_score.append(score)

        img_batch_score = torch.cat(img_batch_score)
        rank_scores.append(img_batch_score.cpu().tolist())
        rank_iids.append(_iid)

    torch.distributed.barrier()
    gather_rank_scores = all_gather(rank_scores)
    gather_rank_iids = all_gather(rank_iids)

    iids = torch.tensor(gather_rank_iids)
    iids = iids.view(-1)
    scores = torch.tensor(gather_rank_scores)
    scores = scores.view(len(iids), -1)

    topk10 = scores.topk(10, dim=1)
    topk5 = scores.topk(5, dim=1)
    topk1 = scores.topk(1, dim=1)
    topk10_iids = tiids[topk10.indices]
    topk5_iids = tiids[topk5.indices]
    topk1_iids = tiids[topk1.indices]

    tr_r10 = (iids.unsqueeze(1) == topk10_iids).float().max(dim=1)[0].mean()
    tr_r5 = (iids.unsqueeze(1) == topk5_iids).float().max(dim=1)[0].mean()
    tr_r1 = (iids.unsqueeze(1) == topk1_iids).float().max(dim=1)[0].mean()

    topk10 = scores.topk(10, dim=0)
    topk5 = scores.topk(5, dim=0)
    topk1 = scores.topk(1, dim=0)
    topk10_iids = iids[topk10.indices]
    topk5_iids = iids[topk5.indices]
    topk1_iids = iids[topk1.indices]

    ir_r10 = (tiids.unsqueeze(0) == topk10_iids).float().max(dim=0)[0].mean()
    ir_r5 = (tiids.unsqueeze(0) == topk5_iids).float().max(dim=0)[0].mean()
    ir_r1 = (tiids.unsqueeze(0) == topk1_iids).float().max(dim=0)[0].mean()

    return (ir_r1, ir_r5, ir_r10, tr_r1, tr_r5, tr_r10)


def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()


def vqa_test_step(pl_module, batch, output):
    id2answer = (
        pl_module.trainer.datamodule.dm_dicts["vqa_trainval"].id2answer
        if "vqa_trainval" in pl_module.trainer.datamodule.dm_dicts
        else pl_module.trainer.datamodule.dm_dicts["vqa"].id2answer
    )
    vqa_logits = output["vqa_logits"]
    vqa_preds = vqa_logits.argmax(dim=-1)
    vqa_preds = [id2answer[pred.item()] for pred in vqa_preds]
    questions = batch["text"]
    qids = batch["qid"]
    return {"qids": qids, "preds": vqa_preds}


def arc_test_step(pl_module, batch, output):
    return output


def vqa_test_wrapup(outs, model_name):
    rank = torch.distributed.get_rank()
    qids, preds = list(), list()
    for out in outs:
        qids += out["qids"]
        preds += out["preds"]

    rets = list()
    for qid, pred in zip(qids, preds):
        rets.append({"question_id": qid, "answer": pred})
    with open(f"vqa_submit_{rank}.json", "w") as fp:
        json.dump(rets, fp, indent=4)

    torch.distributed.barrier()

    if rank == 0:
        jsons = list()
        paths = list(glob.glob("vqa_submit_*.json"))
        for path in paths:
            with open(path, "r") as fp:
                jsons += json.load(fp)
        os.makedirs("result", exist_ok=True)
        with open(f"result/vqa_submit_{model_name}.json", "w") as fp:
            json.dump(jsons, fp, indent=4)

    torch.distributed.barrier()
    os.remove(f"vqa_submit_{rank}.json")


def arc_test_wrapup(outs, caplen, model_name):
    rank = torch.distributed.get_rank()
    iids, captions = list(), list()
    for out in outs:
        iids += out["iid"]
        captions += out["captions"]

    rets = list()
    for iid, caption in zip(iids, captions):
        rets.append({"image_id": iid, "caption": caption})
    with open(f"coco_cap_len{caplen}_{rank}.json", "w") as fp:
        json.dump(rets, fp, indent=4)

    torch.distributed.barrier()

    if rank == 0:
        jsons = list()
        paths = list(glob.glob(f"coco_cap_len{caplen}_*.json"))
        for path in paths:
            with open(path, "r") as fp:
                jsons += json.load(fp)
        os.makedirs("result/arc", exist_ok=True)
        jsons = sorted(jsons, key=lambda x: x["image_id"])
        with open(f"result/arc/coco_cap_{model_name}_len{caplen}.json", "w") as fp:
            json.dump(jsons, fp, indent=4)

    torch.distributed.barrier()
    os.remove(f"coco_cap_len{caplen}_{rank}.json")

def get_current_device():
    '''
    Returns the index of a currently selected device (gpu/cpu).
    '''
    if torch.cuda.is_available():
        return torch.cuda.current_device()
    else:
        return 'cpu'
