import torch
from sklearn.linear_model import LogisticRegression
from config import parse_args
from sklearn.linear_model import LogisticRegression
import numpy as np
from models.only_backbone import OnlyBackboneModel
from models.backbone.Resnet12_pool import resnet12
from torch.nn import CosineSimilarity
import torch.optim as optim

args = parse_args()


def finetune(model, support_imgs):
    print("Finetuning......")
    # fix one image from a single class, 20 negative pairs, total 25 * 20 negative pairs
    # fix one image from a single class, 4 positive paris, total 25 * 4 positive pairs
    # total 600 pairs

    optimizer = optim.AdamW([{"params": model.parameters(), "lr": args.lr, "weight_decay": 0.05}])

    for epoch in range(2):
        for i in range(args.eval_n_way * args.n_shot):
            class_id = int(i / args.n_shot)
            negative_samples = []
            positive_samples = []

            # obtain positive samples
            for k in range(args.n_shot * class_id, args.n_shot * (class_id+1)):
                if k == i: continue
                positive_samples.append(support_imgs[k])  # [args.n_shot-1, 3, 224, 224]

            # obtain negative samples
            for j in range(args.eval_n_way):
                if j == class_id: continue
                negative_samples.append(support_imgs[args.n_shot*j:args.n_shot*(j+1)])  # [args.n_shot * (args.eval_n_way-1), 3, 224, 224]

            positive_samples = torch.stack(positive_samples)
            negative_samples = torch.stack(negative_samples).view(-1, 3, args.image_size, args.image_size)

            positive_bsz = positive_samples.shape[0]
            negative_bsz = negative_samples.shape[0]

            pos_query_img = support_imgs[i].expand(positive_bsz, 3, args.image_size, args.image_size)
            neg_query_img = support_imgs[i].expand(negative_bsz, 3, args.image_size, args.image_size)

            # positive samples
            loss = model(positive_samples, pos_query_img, label=torch.ones(positive_bsz), mode='train', is_feature=args.eval_is_feature)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # negative samples
            loss = model(negative_samples, neg_query_img, label=torch.zeros(negative_bsz), mode='train', is_feature=args.eval_is_feature)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def kshot_test(model, support_imgs, query_imgs, is_feature, method):
    """
    :param support_imgs: [n_way*n_shot, 3, img_size, img_size]
    :param query_imgs:  [1, 3, img_size, img_size]
    :return:
    """
    assert 0 <= method <= 9

    # finetune
    # model.train()
    # finetune(model, support_imgs)
    # model.eval()

    support_labels = np.repeat(range(args.eval_n_way), args.n_shot)
    cos = CosineSimilarity(dim=1, eps=1e-6)

    if method == 0 or method == 1:
        support_feats = []
        query_feats = []  # 25 * [75, dim]
        clf = LogisticRegression(random_state=0, solver='lbfgs', max_iter=1000, penalty='l2',
                                 multi_class='multinomial')
        for j in range(args.n_shot * args.eval_n_way):
            support_img = support_imgs[j].expand(args.n_query * args.eval_n_way, 3, args.image_size, args.image_size)
            support_feat, query_feat = model(support_img=support_img, query_img=query_imgs, label=None, mode="eval", is_feature=is_feature)
            query_feats.append(query_feat)
            support_feats.append(support_feat)

    if method == 0:
        support_feats = torch.mean(torch.stack(support_feats), dim=1).flatten(1)  # [25, 75 * dim]
        support_feats = support_feats.detach().cpu().numpy()
        clf.fit(support_feats, support_labels)
        query_feats = torch.mean(torch.stack(query_feats).permute(1, 0, 2), dim=1).flatten(1)  # [75, 25 * dim]
        query_feats = query_feats.detach().cpu().numpy()
        scores = clf.predict(query_feats)

    elif method == 1:
        support_feats = torch.stack(support_feats).permute(1, 0, 2)  # [75, 25, dim]
        support_feats = support_feats.detach().cpu().numpy()
        for i in range(support_feats.shape[0]):
            print(support_feats[i].shape)
            clf.fit(support_feats[i], support_labels)
        query_feats = torch.stack(query_feats).permute(1, 0, 2)  # [75, 25, dim]
        query_feats = query_feats.detach().cpu().numpy()
        scores = []
        for i in range(query_feats.shape[0]):
            score = clf.predict(query_feats[i])  # [25, 1]
            score = np.argmax(np.bincount(score))
            scores.append(score)
        scores = np.asarray(scores)

    # only use Resnet12_IE
    elif method == 2:
        backbone = resnet12(avg_pool=True, drop_rate=0.1, dropblock_size=5, num_classes=64, no_trans=16, embd_size=64)
        ckpt = torch.load(args.pretrained_path)["model"]
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in ckpt.items():
            name = k.replace("module.", "")
            new_state_dict[name] = v
        backbone.load_state_dict(new_state_dict)
        backbone.cuda()
        backbone.eval()  # turning to eval mode is very important, or results will be very different!

        support_feat = backbone(support_imgs)
        support_feat = support_feat.detach().cpu().numpy()
        # query_feat = backbone(query_imgs)
        # query_feat = query_feat.detach().cpu().numpy()

        clf = LogisticRegression(random_state=0, solver='lbfgs', max_iter=1000, penalty='l2',
                                 multi_class='multinomial')
        clf.fit(support_feat, support_labels)
        # scores = clf.predict(query_feat)

        scores = []
        for i in range(args.eval_n_way * args.n_query):
            cur_query_img = query_imgs[i].expand(1, 3, args.image_size, args.image_size)
            query_feat = backbone(cur_query_img)
            query_feat = query_feat.detach().cpu().numpy()
            score = clf.predict(query_feat)[0]
            scores.append(score)
        scores = np.asarray(scores)

    elif method == 3:  # 最终采用的方式
        scores = []
        for i in range(args.eval_n_way * args.n_query):
            if is_feature:
                cur_query_img = query_imgs[i].expand(args.eval_n_way * args.n_shot, query_imgs.shape[-1])
            else:
                cur_query_img = query_imgs[i].expand(args.eval_n_way*args.n_shot, 3, args.image_size, args.image_size)
            support_feat, query_feat = model(support_img=support_imgs, query_img=cur_query_img, label=None, mode="eval", is_feature=is_feature)
            support_feat = support_feat.detach().cpu().numpy()
            query_feat = torch.mean(query_feat, dim=0, keepdim=True)

            query_feat = query_feat.detach().cpu().numpy()
            clf = LogisticRegression(random_state=0, solver='lbfgs', max_iter=1000, penalty='l2',
                                     multi_class='multinomial')
            clf.fit(support_feat, support_labels)
            score = clf.predict(query_feat)[0]
            scores.append(score)

        scores = np.asarray(scores)

    #
    elif method == 4:
        scores = []
        for i in range(args.eval_n_way * args.n_query):
            if not is_feature:
                cur_query_img = query_imgs[i].expand(args.eval_n_way * args.n_shot, 3, args.image_size, args.image_size)
            else:
                cur_query_img = query_imgs[i].expand(args.eval_n_way * args.n_shot, query_imgs.shape[-1])
            with torch.no_grad():
                support_feat, query_feat = model(support_img=support_imgs, query_img=cur_query_img, label=None, mode="eval",
                                                 is_feature=is_feature)
            support_feats = [torch.mean(support_feat[args.n_shot*j:args.n_shot*(j+1)], dim=0) for j in range(args.eval_n_way)]
            support_feats = torch.stack(support_feats)  #  [args.eval_n_way, hidden_dim]
            query_feat = torch.mean(query_feat, dim=0, keepdim=True).expand(args.eval_n_way, support_feats.shape[1])  # [1, hidden_dim]
            dist = cos(query_feat, support_feats)
            _, top_label = dist.data.topk(1, 0, True, True)
            score = top_label.item()
            scores.append(score)

        scores = np.asarray(scores)

    elif method == 5:  # 39%
        scores = []
        for i in range(args.eval_n_way * args.n_query):
            if not is_feature:
                cur_query_img = query_imgs[i].expand(args.eval_n_way * args.n_shot, 3, args.image_size, args.image_size)
            else:
                cur_query_img = query_imgs[i].expand(args.eval_n_way * args.n_shot, query_imgs.shape[-1])
            with torch.no_grad():
                support_feat, query_feat = model(support_img=support_imgs, query_img=cur_query_img, label=None, mode="eval",
                                             is_feature=is_feature)
            dist = cos(query_feat, support_feat)
            _, top_label = dist.data.topk(1, 0, True, True)
            score = int(top_label.item() / args.n_shot)
            scores.append(score)

    elif method == 6:  # 37.44% +- 1.53%
        scores = []
        for i in range(args.eval_n_way * args.n_query):
            if not is_feature:
                cur_query_img = query_imgs[i].expand(args.eval_n_way * args.n_shot, 3, args.image_size, args.image_size)
            else:
                cur_query_img = query_imgs[i].expand(args.eval_n_way * args.n_shot, query_imgs.shape[-1])
            support_feat, query_feat = model(support_img=support_imgs, query_img=cur_query_img, label=None, mode="eval",
                                             is_feature=is_feature)
            dist = cos(query_feat, support_feat)
            _, top_label = dist.data.topk(1, 0, True, True)
            score = int(top_label.item() / args.n_shot)
            scores.append(score)

    elif method == 7:  # 39.40%
        scores = []
        for i in range(args.eval_n_way * args.n_query):
            if not is_feature:
                cur_query_img = query_imgs[i].expand(args.eval_n_way * args.n_shot, 3, args.image_size, args.image_size)
            else:
                cur_query_img = query_imgs[i].expand(args.eval_n_way * args.n_shot, query_imgs.shape[-1])
            support_feat, query_feat = model(support_img=support_imgs, query_img=cur_query_img, label=None, mode="eval",
                                             is_feature=is_feature)
            dist = cos(query_feat, support_feat)  # [args.eval_n_way * args.n_shots]
            dists = [torch.sum(dist[args.n_shot*j:args.n_shot*(j+1)]) for j in range(args.eval_n_way)]
            dists = torch.stack(dists)
            _, top_label = dists.data.topk(1, 0, True, True)
            score = top_label.item()
            scores.append(score)

    # weighted distance
    elif method == 8:
        weight = torch.tensor([1.0, 0.8, 0.6, 0.4, 0.2])
        weight = weight.cuda()
        scores = []
        for i in range(args.eval_n_way * args.n_query):
            cur_query_img = query_imgs[i].expand(args.eval_n_way * args.n_shot, 3, args.image_size, args.image_size)
            support_feat, query_feat = model(support_img=support_imgs, query_img=cur_query_img, label=None, mode="eval",
                                             is_feature=is_feature)
            dist = cos(query_feat, support_feat)  # [args.eval_n_way * args.n_shots]
            top_5scores, top_5label = dist.data.topk(5, 0, True, True)  # top 5 scores
            weighted_scores = weight * top_5scores
            for i in range(top_5label.shape[0]):
                top_5label[i] = int(top_5label[i] / args.n_shot)
            dists = torch.zeros_like(torch.randn(args.eval_n_way))
            for i in range(args.eval_n_way):
                dists[top_5label[i]] = dists[top_5label[i]] + weighted_scores[i]
            _, score = dists.data.topk(1, 0, True, True)
            score = score.item()
            scores.append(score)

        scores = np.asarray(scores)

    return scores



