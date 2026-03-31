from pymongo import MongoClient
import pandas as pd
import copy
from collections import defaultdict

# ============ 1) Load mapping table ============
def load_api_mapping(mapping_file: str) -> dict:
    """
    Load PyTorch -> MindSpore operator mapping table.
    """
    df = pd.read_csv(mapping_file)
    mapping = {}

    for _, row in df.iterrows():
        pt_api = str(row["PyTorch APIs"]).strip()
        ms_api = str(row["MindSpore APIs"]).strip()
        note = str(row.get("说明", "")).strip()
        mapping[pt_api] = {"ms_api": ms_api, "note": note}

    return mapping



# ============ 2) Type and API conversion helpers ============
def convert_dtype(torch_dtype: str) -> str:
    """
    Convert torch dtype to mindspore dtype.
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
    Convert torch API to MindSpore API.
    Returns (converted_name, method).
    """
    # Prefer mapping table
    if torch_api in mapping:
        note = mapping[torch_api]["note"]
        if "一致" in note:
            return mapping[torch_api]["ms_api"], "Mapping table"
        # If not consistent, fall back to default replacement logic
    # Special cases
    special_mapping = {
        "torch.atleast_3d": "mindspore.ops.atleast_3d",
        "torch.diag_embed": "mindspore.ops.diag_embed",
        "torch.floor_divide": "mindspore.ops.floor_divide",
    }
    if torch_api in special_mapping:
        return special_mapping[torch_api], "Name conversion (special rules)"

    # Default replacement logic
    return torch_api.replace("torch", "mindspore.mint", 1), "Name conversion"


# ============ 3) Record conversion ============
def convert_record(record: dict, mapping: dict) -> dict:
    """
    Convert all fields for a single test case.
    """
    new_record = {}
    for key, value in record.items():
        if key == "_id":
            continue  # Skip old _id

        # Handle API name
        if key == "api":
            new_record[key], _ = convert_api_name(value, mapping)
            continue

        # Handle dtype
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


# ============ 4) Main flow ============
if __name__ == "__main__":
    # 1) Load mapping table
    mapping_file = "D:/graduate/testcasegen/api_mapping/pt_ms_mapping.csv"
    api_mapping = load_api_mapping(mapping_file)
    print(f"--- Mapping table loaded: {len(api_mapping)} entries")

    # 2) Connect MongoDB
    client = MongoClient("mongodb://localhost:27017/")
    db = client["freefuzz-torch"]
    print("--- Database connected")

    src_collection = db["argVS"]
    dst_db = client["new-mindspore"]
    dst_collection = dst_db["ms-cases"]

    # Clear target collection
    dst_collection.delete_many({})
    print("--- Cleared new-mindspore.ms-cases collection")

    # 3) Convert by operator
    api_groups = defaultdict(list)
    for record in src_collection.find():
        api_name = record.get("api", "")
        api_groups[api_name].append(record)

    total_count = 0
    map_count = 0  # Operators converted using mapping table
    name_count = 0  # Operators converted using name conversion

    for api_name, records in api_groups.items():
        converted_api, method = convert_api_name(api_name, api_mapping)
        if method == "Mapping table":
            map_count += 1
        else:
            name_count += 1

        for record in records:
            new_record = convert_record(record, api_mapping)
            dst_collection.insert_one(new_record)
            total_count += 1

        print(f"✔ {api_name} converted with method: {method}, total records: {len(records)}")

    print(f"--- Total records processed: {total_count} ---")
    print(f"--- Operators converted via mapping table: {map_count} ---")
    print(f"--- Operators converted via name conversion: {name_count} ---")
