import torch
from sklearn.linear_model import LogisticRegression
from config import parse_args
import numpy as np

args = parse_args()


def kshot_test(model, support_imgs, query_imgs):
    """
    :param support_imgs: [n_way*n_shot, 3, img_size, img_size]
    :param query_imgs:  [1, 3, img_size, img_size]
    :return:
    """

    support_labels = np.repeat(range(args.eval_n_way), args.n_shot)
    scores = []
    for i in range(args.eval_n_way * args.n_query):
        cur_query_img = query_imgs[i].expand(args.eval_n_way*args.n_shot, 3, args.image_size, args.image_size)

        support_feat, query_feat = model(support_img=support_imgs, query_img=cur_query_img, label=None, split="eval")
        support_feat = support_feat.detach().cpu().numpy()

        query_feat = torch.mean(query_feat, dim=0, keepdim=True)
        query_feat = query_feat.detach().cpu().numpy()

        clf = LogisticRegression(random_state=0, solver='lbfgs', max_iter=1000, penalty='l2',
                                    multi_class='multinomial')
        clf.fit(support_feat, support_labels)
        score = clf.predict(query_feat)[0]
        scores.append(score)

    scores = np.asarray(scores)

    return scores



