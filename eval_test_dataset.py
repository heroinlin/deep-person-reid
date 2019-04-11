
import torch
from torch import nn
from torchreid import models
import numpy as np
from collections import OrderedDict
from torchvision import transforms
import cv2
from pathlib import Path
from datetime import datetime
from data import ODData
from collections import defaultdict
import pickle


def read_label(boxes, image_width, image_height):
    for box in boxes:
        x1, y1, x2, y2 = box
        x, y, w, h = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
        # 框扩大1.5倍
        w = min(w * 1.5, 1.0)
        h = min(h * 1.5, 1.0)
        x1, y1, x2, y2 = x - w / 2, y - h / 2, x + w / 2, y + h / 2
        # 到图像范围
        x1, y1, x2, y2 = round(x1 * image_width), round(y1 * image_height), round(x2 * image_width), round(y2 * image_height)
        x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, image_width), min(y2, image_height)
        box = [x1, y1, x2, y2]
        return box


def normalize(nparray, order=2, axis=0):
    """Normalize a N-D numpy array along the specified axis."""
    norm = np.linalg.norm(nparray, ord=order, axis=axis, keepdims=True)
    return nparray / (norm + np.finfo(np.float32).eps)


def compute_dist(array1, array2, type='euclidean'):
    """Compute the euclidean or cosine distance of all pairs.
    Args:
      array1: numpy array with shape [m1, n]
      array2: numpy array with shape [m2, n]
      type: one of ['cosine', 'euclidean']
    Returns:
      numpy array with shape [m1, m2]
    """
    assert type in ['cosine', 'euclidean']
    if type == 'cosine':
        array1 = normalize(array1, axis=1)
        array2 = normalize(array2, axis=1)
        dist = np.matmul(array1, array2.T)
        return dist
    else:
        # shape [m1, 1]
        square1 = np.sum(np.square(array1), axis=1)[..., np.newaxis]
        # shape [1, m2]
        square2 = np.sum(np.square(array2), axis=1)[np.newaxis, ...]
        squared_dist = - 2 * np.matmul(array1, array2.T) + square1 + square2
        squared_dist[squared_dist < 0] = 0
        dist = np.sqrt(squared_dist)
        return dist


def re_ranking(q_g_dist, q_q_dist, g_g_dist, k1=20, k2=6, lambda_value=0.3):
    # The following naming, e.g. gallery_num, is different from outer scope.
    # Don't care about it.

    original_dist = np.concatenate(
        [np.concatenate([q_q_dist, q_g_dist], axis=1),
         np.concatenate([q_g_dist.T, g_g_dist], axis=1)],
        axis=0)
    original_dist = np.power(original_dist, 2).astype(np.float32)
    original_dist = np.transpose(1. * original_dist / np.max(original_dist, axis=0))
    V = np.zeros_like(original_dist).astype(np.float32)
    initial_rank = np.argsort(original_dist).astype(np.int32)

    query_num = q_g_dist.shape[0]
    gallery_num = q_g_dist.shape[0] + q_g_dist.shape[1]
    all_num = gallery_num

    for i in range(all_num):
        # k-reciprocal neighbors
        forward_k_neigh_index = initial_rank[i, :k1 + 1]
        backward_k_neigh_index = initial_rank[forward_k_neigh_index, :k1 + 1]
        fi = np.where(backward_k_neigh_index == i)[0]
        k_reciprocal_index = forward_k_neigh_index[fi]
        k_reciprocal_expansion_index = k_reciprocal_index
        for j in range(len(k_reciprocal_index)):
            candidate = k_reciprocal_index[j]
            candidate_forward_k_neigh_index = initial_rank[candidate, :int(np.around(k1 / 2.)) + 1]
            candidate_backward_k_neigh_index = initial_rank[candidate_forward_k_neigh_index,
                                               :int(np.around(k1 / 2.)) + 1]
            fi_candidate = np.where(candidate_backward_k_neigh_index == candidate)[0]
            candidate_k_reciprocal_index = candidate_forward_k_neigh_index[fi_candidate]
            if len(np.intersect1d(candidate_k_reciprocal_index, k_reciprocal_index)) > 2. / 3 * len(
                    candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index, candidate_k_reciprocal_index)

        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
        weight = np.exp(-original_dist[i, k_reciprocal_expansion_index])
        V[i, k_reciprocal_expansion_index] = 1. * weight / np.sum(weight)
    original_dist = original_dist[:query_num, ]
    if k2 != 1:
        V_qe = np.zeros_like(V, dtype=np.float32)
        for i in range(all_num):
            V_qe[i, :] = np.mean(V[initial_rank[i, :k2], :], axis=0)
        V = V_qe
        del V_qe
    del initial_rank
    invIndex = []
    for i in range(gallery_num):
        invIndex.append(np.where(V[:, i] != 0)[0])

    jaccard_dist = np.zeros_like(original_dist, dtype=np.float32)

    for i in range(query_num):
        temp_min = np.zeros(shape=[1, gallery_num], dtype=np.float32)
        indNonZero = np.where(V[i, :] != 0)[0]
        indImages = []
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0, indImages[j]] = temp_min[0, indImages[j]] + np.minimum(V[i, indNonZero[j]],
                                                                               V[indImages[j], indNonZero[j]])
        jaccard_dist[i] = 1 - temp_min / (2. - temp_min)

    final_dist = jaccard_dist * (1 - lambda_value) + original_dist * lambda_value
    del original_dist
    del V
    del jaccard_dist
    final_dist = final_dist[:query_num, query_num:]
    return final_dist


class CTest:
    def __init__(self, schedule: tuple, schedule_data: dict, model, batch_size: int, device="cpu"):
        super(CTest, self).__init__()
        self.model = model.to(device)
        self.device = device

        self.schedule = schedule
        self.schedule_data = schedule_data

        self.batch_size = batch_size

        # const
        self.image_width = 144
        self.image_height = 144
        self.mean = [0.485, 0.456, 0.406]
        self.stddev = [0.229, 0.224, 0.225]
        self.feature_size = 1024
        self.max_time_gap = 7200

        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.stddev)
        ])
        self.features = [{}, {}]
        self.times = [[], []]

    def image_process(self, image_file_path, box, direction):
        image = cv2.imread(str(image_file_path))
        image = image[box[1]:box[3], box[0]:box[2], :]
        if direction == 1:
            image = cv2.flip(image, -1)
        image = cv2.resize(image, dsize=(self.image_width, self.image_height))
        # # 图像随机灰度化
        # grayCode = random.randint(0, 1)
        # image = random_gray(image, grayCode)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_array = (np.array(image, dtype=np.float32)/255 - self.mean)/self.stddev
        return img_array

    # 读取图片路径列表下的所有图片，组成一个batch_images 的np数组
    def read_batch(self, image_list, direction):
        batch_images = []
        for image_item in image_list:
            image = self.image_process(image_item["image_path"], image_item["box"], direction)
            batch_images.append(image)
        np.vstack(batch_images)
        return batch_images

    def batch_image_test(self, batch_images):
        batch_images = np.transpose(batch_images, [0, 3, 1, 2])
        batch_images = torch.from_numpy(batch_images).float().to(self.device)
        features = self.model(batch_images)
        features_array = features.data.cpu().numpy()
        return features_array

    def feature_extract(self, direction):
        class_dict = self.schedule_data[direction]
        person_features = OrderedDict()
        image_times = list()
        for person_id, image_list in class_dict.items():
            print("{}_person_id: {}, have {} images".format("gallery" if direction == 0 else "query", person_id, len(image_list)))
            image_time = image_list[0]["time"]
            batch_num = len(image_list) // self.batch_size
            person_id_features = np.zeros([0, self.feature_size], dtype=np.float)
            for index in range(batch_num):
                batch_images = self.read_batch(image_list[index*self.batch_size: (index+1)*self.batch_size], direction)
                features_array = self.batch_image_test(batch_images)
                person_id_features = np.vstack((person_id_features, features_array))
            if len(image_list) % self.batch_size:
                batch_images = self.read_batch(image_list[batch_num*self.batch_size::], direction)
                features_array = self.batch_image_test(batch_images)
                person_id_features = np.vstack((person_id_features, features_array))
            # 取特征的均值
            person_id_features = np.mean(person_id_features, axis=0)
            # 归一化
            sq_sum = 1 / np.sqrt(np.sum(np.square(person_id_features)) + 1e-6)
            sq_sum = np.array([sq_sum])
            person_id_features = person_id_features * sq_sum

            person_features[person_id] = person_id_features
            image_times.append(image_time)
        self.features[direction] = person_features
        self.times[direction] = image_times

    def compute_diff_mat(self) -> np.array:
        gallery_features, query_features = self.features
        gallery_times, query_times = self.times
        q_feat = np.zeros((len(query_features), self.feature_size), dtype=np.float)
        # q_feat = np.zeros((len(self.query_features), 618), dtype=np.float)
        q_label = []
        for index, (person_id, _) in enumerate(query_features.items()):
            q_feat[index, :] = query_features[person_id]
            # q_feat[index, :] = self.query_features[person_id][np.where(self.query_features[person_id] > 10e-4)]
            q_label.append(person_id)
        # 根据对应查询的库ID，提取距离矩阵中对应的行，用于预测对应库ID
        g_feat = np.zeros((len(gallery_features), self.feature_size), dtype=np.float)
        # g_feat = np.zeros((len(self.gallery_features), 618), dtype=np.float)
        g_label = []
        for index, (person_id, _) in enumerate(gallery_features.items()):
            g_feat[index, :] = gallery_features[person_id]
            # g_feat[index, :] = self.gallery_features[person_id][np.where(self.gallery_features[person_id] > 10e-4)]
            g_label.append(person_id)
        dist_mat = 1 - np.dot(q_feat, g_feat.transpose())
        query_persons_num = dist_mat.shape[0]
        gallery_persons_num = dist_mat.shape[1]
        for q in range(query_persons_num):
            for g in range(gallery_persons_num):
                if self._date_time_compare(query_times[q], gallery_times[g]) < 0:  # q_times[q] <= g_times[g]:
                    dist_mat[q, g] = 1000
        # distmat_file = open("distmat.plk", "wb")
        # pickle.dump(distmat, distmat_file, protocol=2)
        # distmat_file.close()
        cmc, mAP = self.eval_Map(dist_mat, q_label, g_label, max_rank=50)
        print("Results ----------")
        print("mAP: {:.1%}".format(mAP))
        print("CMC curve")
        for r in [1, 3, 5, 10]:
            print("Rank-{:<3}: {:.1%}".format(r, cmc[r - 1]))
        print("------------------")
        result_set = self.iterate_algorithm(dist_mat, q_label)
        # result_set = self.conditions_eval(dist_mat, q_label, margin=1.2)
        for q in range(len(q_label)):
            # result_set[q, 1] = g_label[q_label.index(result_set[q, 0])]  # 将查询样本ID对应的正确库样本ID计算出来
            result_set[q, 3] = g_label[result_set[q, 2]]
        return result_set

    def re_ranking(self) -> np.array:
        gallery_features, query_features = self.features
        gallery_times, query_times = self.times
        g_label = list(np.vstack(gallery_features.keys()))
        q_label = list(np.vstack(query_features.keys()))
        g_g_dist = compute_dist(np.vstack(gallery_features.values()),
                                np.vstack(gallery_features.values()), type='euclidean')
        # gallery-gallery distance
        q_q_dist = compute_dist(np.vstack(query_features.values()), np.vstack(query_features.values()),
                                type='euclidean')
        # re-ranked query-gallery distance
        q_g_dist = compute_dist(np.vstack(query_features.values()), np.vstack(gallery_features.values()),
                                type='euclidean')
        re_r_q_g_dist = re_ranking(q_g_dist, q_q_dist, g_g_dist)
        query_persons_num = re_r_q_g_dist.shape[0]
        gallery_persons_num = re_r_q_g_dist.shape[1]
        for q in range(query_persons_num):
            for g in range(gallery_persons_num):
                if self._date_time_compare(query_times[q], gallery_times[g]) < 0:  # q_times[q] <= g_times[g]:
                    re_r_q_g_dist[q, g] = 1000
        result_set = self.iterate_algorithm(re_r_q_g_dist, q_label)
        # result_set = self.conditions_eval(re_r_q_g_dist, q_label)
        for q in range(len(q_label)):
            result_set[q, 1] = g_label[q_label.index(result_set[q, 0])]  # 将查询样本ID对应的正确库样本ID计算出来
            result_set[q, 3] = g_label[result_set[q, 2]]
        return result_set

    def conditions_eval(self, distmat, new_qlabel, max_rank=5, margin=0.3):
        indices = np.argsort(distmat, axis=1)
        distmat_sort = np.array([distmat[i, indices[:, 0:max_rank][i]] for i in range(len(indices[:, 0]))])
        # distmat_sort = distmat_sort / np.tile(np.sum(distmat_sort, axis=1), [max_rank, 1]).transpose()
        margin_condition = distmat_sort[:, 0] < margin
        compare_condition = (distmat_sort[:, 1] - distmat_sort[:, 0]) / (distmat_sort[:, 1] + 0.0000001) > 0.1
        all_conditions = margin_condition & compare_condition
        q_len = distmat.shape[0]
        res_set = np.zeros((q_len, 4), dtype=np.int)
        for index, condition in enumerate(all_conditions):
            res_set[index, 0] = new_qlabel[index]
            res_set[index, 1] = res_set[index, 0]  # 存储查询样本对应的正确库样本的编号
            if condition:
                res_set[index, 2] = indices[index, 0]
            else:
                res_set[index, 2] = -1
        return res_set

    def eval_Map(self, distmat, q_pids, g_pids, max_rank):
        """Evaluation with cuhk03 metric
        Key: one image for each gallery identity is randomly sampled for each query identity.
        Random sampling is performed num_repeats times.
        """
        num_repeats = 10
        num_q, num_g = distmat.shape
        q_pids = np.asarray(q_pids)
        g_pids = np.asarray(g_pids)
        if num_g < max_rank:
            max_rank = num_g
            print("Note: number of gallery samples is quite small, got {}".format(num_g))

        indices = np.argsort(distmat, axis=1)
        matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

        # compute cmc curve for each query
        all_cmc = []
        all_AP = []
        num_valid_q = 0.  # number of valid query

        for q_idx in range(num_q):
            # remove gallery samples that have the same pid and camid with query
            order = indices[q_idx]

            # compute cmc curve
            raw_cmc = matches[q_idx]  # binary vector, positions with value 1 are correct matches
            if not np.any(raw_cmc):
                # this condition is true when query identity does not appear in gallery
                continue

            kept_g_pids = g_pids[order]
            g_pids_dict = defaultdict(list)
            for idx, pid in enumerate(kept_g_pids):
                g_pids_dict[pid].append(idx)

            cmc, AP = 0., 0.
            for repeat_idx in range(num_repeats):
                mask = np.zeros(len(raw_cmc), dtype=np.bool)
                for _, idxs in g_pids_dict.items():
                    # randomly sample one image for each gallery person
                    rnd_idx = np.random.choice(idxs)
                    mask[rnd_idx] = True
                masked_raw_cmc = raw_cmc[mask]
                _cmc = masked_raw_cmc.cumsum()
                _cmc[_cmc > 1] = 1
                cmc += _cmc[:max_rank].astype(np.float32)
                # compute AP
                num_rel = masked_raw_cmc.sum()
                tmp_cmc = masked_raw_cmc.cumsum()
                tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
                tmp_cmc = np.asarray(tmp_cmc) * masked_raw_cmc
                AP += tmp_cmc.sum() / num_rel

            cmc /= num_repeats
            AP /= num_repeats
            all_cmc.append(cmc)
            all_AP.append(AP)
            num_valid_q += 1.

        assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

        all_cmc = np.asarray(all_cmc).astype(np.float32)
        all_cmc = all_cmc.sum(0) / num_valid_q
        mAP = np.mean(all_AP)

        return all_cmc, mAP

    def write_result_to_csv(self, result_set):
        # 根据真实值和预测值进行准确度计算
        print(result_set)
        result_dir = Path("test_result")
        result_dir.mkdir(exist_ok=True)
        vv = 0  # 预测结果正确计数器
        for v in range(result_set.shape[0]):  # 遍历所有乘客
            if result_set[v, 1] == result_set[v, 3]:  # 预测结果正确
                vv = vv + 1
        prec = vv / float(result_set.shape[0])
        # 将一个班次的真实结果和预测结果保存到文件
        bus_schedules = "-".join(self.schedule)
        with open(result_dir / f'{bus_schedules}_result.csv', 'w') as fin:
            fin.write('Truth queryID, Truth galleryID,Pred galleryIDIdx,Pred galleryID\n')
            for ii in range(result_set.shape[0]):
                fin.write('%d,%d,%d,%d\n' % (result_set[ii, 0], result_set[ii, 1], result_set[ii, 2], result_set[ii, 3]))
            fin.write('The total prec:, %f, Passager num:, %d\n' % (prec, len(result_set)))
        with open(result_dir / 'total_result.csv', 'a') as total_fin:
            total_fin.write(
                'bus_schedules:, %s, The total prec:, %f, Passager num:, %d\n' % (bus_schedules, prec, len(result_set)))

    @staticmethod
    def iterate_predict_next(res_set, dist_sorted, dist_sorted_idx):
        q_len = dist_sorted.shape[0]  # 查询乘客个数
        g_len = dist_sorted.shape[1]  # 库乘客个数
        cnt = 0
        match_set = []  # 统计待预测的乘客编号
        for q in range(q_len):
            if res_set[q, 2] > -1:
                cnt = cnt + 1
                match_set.append(res_set[q, 2])

        dis_mat_s = np.zeros((q_len, g_len), dtype=np.float)
        idx_mat_s = np.zeros((q_len, g_len), dtype=np.float)
        for q in range(q_len):
            if res_set[q, 2] == -1:
                j = 0
                for g in range(g_len):
                    if dist_sorted_idx[q, g] not in match_set:
                        dis_mat_s[q, j] = dist_sorted[q, g]
                        idx_mat_s[q, j] = dist_sorted_idx[q, g]
                        j = j + 1

        if q_len - len(match_set) == 1:
            for q in range(q_len):
                if res_set[q, 2] == -1:
                    return q, idx_mat_s[q, 0]

        thresh_mat = (dis_mat_s[:, 1] - dis_mat_s[:, 0]) / (dis_mat_s[:, 1] + 0.000000001)

        maxVal = 0
        maxIdx = -1
        for q in range(q_len):
            if thresh_mat[q] > maxVal:
                maxVal = thresh_mat[q]
                maxIdx = q
        return maxIdx, idx_mat_s[maxIdx, 0]

    # 核心预测算法：迭代预测算法！！！
    def iterate_algorithm(self, dist_mat, new_qlabel):
        dist_sorted = np.sort(dist_mat, axis=1)
        dist_sorted_idx = np.argsort(dist_mat, axis=1)
        q_len = dist_mat.shape[0]

        thresh_mat = (dist_sorted[:, 1] - dist_sorted[:, 0]) / (dist_sorted[:, 1] + 0.000000001)

        # res_set为结果集：第一列为真实标签，第二列为预测标签
        res_set = np.zeros((q_len, 4), dtype=np.int)
        for q in range(q_len):
            res_set[q, 0] = new_qlabel[q]
            res_set[q, 1] = res_set[q, 0]  # 存储查询样本对应的正确库样本的编号
            res_set[q, 2] = -1  # 初始均为待定
            if thresh_mat[q] > 0.2:
                res_set[q, 2] = dist_sorted_idx[q, 0]

        # 如果有重复标签，则全部设为待预测
        for q1 in range(q_len):
            for q2 in range(q_len):
                if q2 > q1 and res_set[q2, 2] == res_set[q1, 2]:
                    if dist_sorted[q1, 0] > dist_sorted[q2, 0]:
                        res_set[q1, 2] = -1
                    else:
                        res_set[q2, 2] = -1
                    # res_set[q1, 2] = -1
                    # res_set[q2, 2] = -1

        # 核心预测过程：迭代预测
        for q in range(q_len):
            if -1 in res_set:
                match_index, match_passager_idx = self.iterate_predict_next(res_set, dist_sorted, dist_sorted_idx)
                res_set[match_index, 2] = match_passager_idx
            else:
                break
        return res_set

    def _date_time_compare(self, query_image_time: datetime, gallery_image_time: datetime):
        if 0 <= (query_image_time - gallery_image_time).total_seconds() <= self.max_time_gap:  # 两小时内
            return 1
        return -1


def test(args):
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(args.checkpoint, map_location=device)

    oddata = ODData(args.annotation_file, args.image_root)
    test_data = oddata.for_test()

    model = models.init_model(name=args.arch, num_classes=6090, loss={'xent', 'htri'}, pretrained=False)
    model = torch.nn.DataParallel(model).to(device) if device == f"cuda:0" else model
    model = model.module if device == f"cuda:0" else model
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    total_correct_person_num = 0
    total_query_person_num = 0

    for schedule, schedule_data in test_data.items():
        print(schedule)
        tester = CTest(schedule, schedule_data, model, args.batch_size, device)
        tester.feature_extract(0)
        tester.feature_extract(1)
        print("gallery have {} persons, query have {} persons".format(*map(len, tester.features)))
        # tester.compute_diff_mat()
        result_set = tester.compute_diff_mat()
        # result_set = tester.re_ranking()
        tester.write_result_to_csv(result_set)
        print("This bus schedules predict accuracy is {:.4f}\n"
              .format((result_set[:, 1] == result_set[:, 3]).sum() / len(result_set)))
        total_correct_person_num += (result_set[:, 1] == result_set[:, 3]).sum()
        total_query_person_num += len(result_set)
    print("total predict accuracy is {:.4f}\n".format(total_correct_person_num / total_query_person_num))
