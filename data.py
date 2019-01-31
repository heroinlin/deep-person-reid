from typing import List, Dict
from pathlib import Path
import pickle
from datetime import datetime
import pytz


def from_timestr(timestr):
    """
    把字符串时间转成datetime格式

    :param timestr: “年_月_日_时_分_秒” 的格式
    :return: datetime
    """
    time = datetime.strptime(timestr, "%Y_%m_%d_%H_%M_%S")
    time = time.astimezone(pytz.timezone("Asia/Shanghai"))
    return time


def read_label(box, image_width, image_height):
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


class ODData(object):
    def __init__(self, annotation_file, image_root):
        self.annotation_file = Path(annotation_file)
        self.image_root = Path(image_root)

    def for_train(self) -> (List[dict], dict):
        """
        获取训练用的数据格式

        :return: (list, dict)
            [
                {
                    ""
                },
                ...
            ]
        """
        annotations = self._read_annotation()
        data_list = []
        data_dict = {}

        current_person_index = 0
        current_person_id = -1
        for annotation in annotations:
            image_path = self.image_root / annotation["image"]["information"]["path"]
            for person in annotation["annotation"]["persons"]:
                person_id = person["person_id"]
                direction = person["direction"]
                box = read_label(person["box"], annotation["image"]["information"]["width"], annotation["image"]["information"]["height"])

                if person_id != current_person_id:
                    current_person_index += 1
                    current_person_id = person_id

                person_class = current_person_index

                data_list.append({
                    "image_path": image_path,
                    "direction": direction,
                    "box": box,
                    "class": person_class
                })
                person_dict = data_dict.setdefault(person_class, {0: [], 1: []})
                person_dict[direction].append({
                    "image_path": image_path,
                    "box": box,
                    "class": person_class
                })

        return data_list, data_dict, current_person_index

    def for_test(self, with_schedule=False) -> Dict[tuple, Dict[int, Dict[int, dict]]]:
        """
        获取测试用的数据格式

        :param with_schedule: 是否加入班次信息
        :return: dict
            {
                tuple(车辆名, 日期[, 班次开始时间, 班次结束时间]): {
                    0: {  // 上车
                        person_id: [  // 包含这个人的多张图片
                            {
                                "image_path": 图片路径,
                                "time": 时间,  // datetime格式，时区为Asia/Shanghai
                                "box": [x1, y1, x2, y2]
                            },
                            ...
                        ],
                        ...
                    },
                    1: {  // 下车
                        person_id: [  // 包含这个人的多张图片
                            {
                                "image_path": 图片路径,
                                "time": 时间,  // datetime格式，时区为Asia/Shanghai
                                "box": [x1, y1, x2, y2]
                            },
                            ...
                        ],
                        ...
                    }
                },
                ...
            }
        """
        annotations = self._read_annotation()

        result = {}
        for annotation in annotations:
            image_path = self.image_root / annotation["image"]["information"]["path"]
            bus_name = annotation["source"]["bus_id"]
            date = annotation["source"]["date"]
            day_time = annotation["source"]["time"]
            time = from_timestr(f"{date}_{day_time}")
            if with_schedule:
                schedule_begin_time = annotation["source"]["schedule_begin_time"]
                schedule_end_time = annotation["source"]["schedule_end_time"]
                schedule_info = bus_name, date, schedule_begin_time, schedule_end_time
            else:
                schedule_info = bus_name, date

            for person in annotation["annotation"]["persons"]:
                person_id = person["person_id"]
                direction = person["direction"]
                box = read_label(person["box"], annotation["image"]["information"]["width"], annotation["image"]["information"]["height"])
                schedule_dict = result.setdefault(schedule_info, {0: {}, 1: {}})
                person_pics = schedule_dict[direction].setdefault(person_id, [])
                person_pics.append({
                    "image_path": image_path,
                    "time": time,
                    "box": box
                })
        return result

    def _read_annotation(self):
        with open(self.annotation_file, "rb") as file:
            return pickle.load(file)
