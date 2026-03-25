from pymongo import MongoClient
import pandas as pd
import copy
from collections import defaultdict

# ============ 一、加载映射表 ============
def load_api_mapping(mapping_file: str) -> dict:
    """
    加载 PyTorch -> MindSpore 算子映射表
    """
    df = pd.read_csv(mapping_file)
    mapping = {}

    for _, row in df.iterrows():
        pt_api = str(row["PyTorch APIs"]).strip()
        ms_api = str(row["MindSpore APIs"]).strip()
        note = str(row.get("说明", "")).strip()
        mapping[pt_api] = {"ms_api": ms_api, "note": note}

    return mapping



# ============ 二、类型与API转换函数 ============
def convert_dtype(torch_dtype: str) -> str:
    """
    将 torch 的 dtype 转换为 mindspore 的 dtype
    """
    mapping = {
        "torch.float32": "mindspore.float32",
        "torch.float64": "mindspore.float64",
        "torch.float16": "mindspore.float16",
        "torch.bfloat16": "mindspore.bfloat16",
        "torch.int64": "mindspore.int64",
        "torch.int32": "mindspore.int32",
        "torch.int16": "mindspore.int16",
        "torch.int8": "mindspore.int8",
        "torch.uint8": "mindspore.uint8",
        "torch.bool": "mindspore.bool_",
    }
    return mapping.get(torch_dtype, torch_dtype.replace("torch.", "mindspore."))


def convert_api_name(torch_api: str, mapping: dict) -> tuple[str, str]:
    """
    将 torch API 转换为 MindSpore API
    返回 (转换后的名称, 使用的方法)
    """
    # 优先查映射表
    if torch_api in mapping:
        note = mapping[torch_api]["note"]
        if "一致" in note:
            return mapping[torch_api]["ms_api"], "映射表"
        # 若非一致，则继续使用默认替换逻辑
    # 特殊情况处理
    special_mapping = {
        "torch.atleast_3d": "mindspore.ops.atleast_3d",
        "torch.diag_embed": "mindspore.ops.diag_embed",
        "torch.floor_divide": "mindspore.ops.floor_divide",
    }
    if torch_api in special_mapping:
        return special_mapping[torch_api], "名称转换(特殊规则)"

    # 默认替换逻辑
    return torch_api.replace("torch", "mindspore.mint", 1), "名称转换"


# ============ 三、记录转换 ============
def convert_record(record: dict, mapping: dict) -> dict:
    """
    转换单个测试用例的所有字段
    """
    new_record = {}
    for key, value in record.items():
        if key == "_id":
            continue  # 跳过旧的 _id

        # 处理 API 名
        if key == "api":
            new_record[key], _ = convert_api_name(value, mapping)
            continue

        # 处理 dtype
        if isinstance(value, list):
            new_list = []
            for item in value:
                if isinstance(item, dict) and "dtype" in item:
                    item = copy.deepcopy(item)
                    item["dtype"] = convert_dtype(item["dtype"])
                new_list.append(item)
            new_record[key] = new_list
        elif isinstance(value, dict):
            new_value = copy.deepcopy(value)
            if "dtype" in new_value:
                new_value["dtype"] = convert_dtype(new_value["dtype"])
            new_record[key] = new_value
        else:
            new_record[key] = value

    return new_record


# ============ 四、主流程 ============
if __name__ == "__main__":
    # 1. 加载映射表
    mapping_file = "D:/graduate/testcasegen/api_mapping/pt_ms_mapping.csv"
    api_mapping = load_api_mapping(mapping_file)
    print(f"--- 映射表加载完成，共 {len(api_mapping)} 条映射")

    # 2. 连接 MongoDB
    client = MongoClient("mongodb://localhost:27017/")
    db = client["freefuzz-torch"]
    print("--- 数据库连接成功")

    src_collection = db["argVS"]
    dst_db = client["new-mindspore"]
    dst_collection = dst_db["ms-cases"]

    # 清空目标集合
    dst_collection.delete_many({})
    print("--- 已清空 new-mindspore.ms-cases 集合")

    # 3. 逐算子转换
    api_groups = defaultdict(list)
    for record in src_collection.find():
        api_name = record.get("api", "")
        api_groups[api_name].append(record)

    total_count = 0
    map_count = 0  # 使用映射表转换的算子数
    name_count = 0  # 使用名称转换的算子数

    for api_name, records in api_groups.items():
        converted_api, method = convert_api_name(api_name, api_mapping)
        if method == "映射表":
            map_count += 1
        else:
            name_count += 1

        for record in records:
            new_record = convert_record(record, api_mapping)
            dst_collection.insert_one(new_record)
            total_count += 1

        print(f"✔ {api_name} 算子已转换完成，使用的方法：{method}，共 {len(records)} 条记录")

    print(f"--- 总共处理完成 {total_count} 条记录 ---")
    print(f"--- 其中通过映射表转换的算子：{map_count} 个 ---")
    print(f"--- 其中通过名称转换的算子：{name_count} 个 ---")
