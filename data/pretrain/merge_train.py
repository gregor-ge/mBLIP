import json
import os

def merge(root, files, out):

    data = []
    for file in files:
        d = json.load(open(os.path.join(root, file), "r", encoding="utf-8"))
        print(file, len(d))
        num_images = len(set([e["image_id"] for e in d]))
        print("Images", num_images)
        data.extend(d)
    print("Total ", len(data))
    with open(out, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    root = ".."
    files = ["pretrain/ccs_synthetic_filtered_large_2273005_mt.json",
             "mscoco/coco_train_mt.json",
             "llava/detail_23k_mt.json",
             "llava/conversation_58k_split_max3sent_mt.json",
             "aokvqa/aokvqa_cot_en.json",
             "aokvqa/aokvqa_explain_en.json",
             "vqa/vqav2_vqa_mt.json",
             "vqa/vqav2_vqg_mt.json"]
    out = "task_mix_v1_mt.json"

    merge(root, files, out)