# 修改代码以支持生成覆盖率元数据

## 问题
`ObjectCategoryMapping` 强制要求 `coverage_meta_file` 参数，但在生成覆盖率元数据时，这个文件还不存在。

## 解决方案
修改 `ovon/dataset/semantic_utils.py` 的 `load_categories` 方法，允许 `coverage_meta_file` 为 None。

## 修改步骤

### 1. 修改 `ovon/dataset/semantic_utils.py:48-111`

```python
@staticmethod
def load_categories(
    mapping_file: str,
    coverage_meta_file: str,  # ← 改为 Optional[str]
    frame_coverage_threshold: float,
    filter_attributes: Optional[Set[str]] = [
        "ceiling",
        "door",
        "floor",
        "object ",
        "wall",
        "unknown",
        "device",
        "decoration",
    ],
) -> Dict[str, str]:
    # ===== 添加这段代码 =====
    # 如果没有提供覆盖率文件，跳过覆盖率过滤
    if coverage_meta_file is None:
        print("Warning: coverage_meta_file is None, skipping coverage filter")

        mapping = {}
        attr_filtering = 0

        with open(mapping_file, "r") as tsv_file:
            tsv_reader = csv.reader(tsv_file, delimiter="\t")
            is_first_row = True
            for row in tsv_reader:
                if is_first_row:
                    is_first_row = False
                    continue
                raw_name = row[1]
                raw_cat_name = row[2]

                # 只应用属性过滤
                ignore_category = False
                for attribute in filter_attributes:
                    if attribute in raw_cat_name.lower():
                        ignore_category = True
                        attr_filtering += 1
                        break

                if "otherroom" in raw_cat_name.lower():
                    raw_cat_name = raw_cat_name.split("/")[0].strip()

                if ignore_category:
                    continue
                mapping[raw_name] = raw_cat_name

        print(
            "Post filtering stats (no coverage filter) - "
            "Ignore category: {}, Final: {}".format(
                attr_filtering, len(mapping.keys())
            )
        )

        return mapping
    # ===== 添加结束 =====

    # 原有代码保持不变
    # Filter based on coverage
    file = open(coverage_meta_file, "rb")
    coverage_metadata = pickle.load(file)

    coverage_metadata_dict = defaultdict(list)
    for category, coverage_meta in coverage_metadata.items():
        for frame_coverage, _, scene in coverage_meta:
            if frame_coverage >= frame_coverage_threshold:
                coverage_metadata_dict[category].append(frame_coverage)

    mapping = {}
    threshold_filtering = 0
    attr_filtering = 0
    with open(mapping_file, "r") as tsv_file:
        tsv_reader = csv.reader(tsv_file, delimiter="\t")
        is_first_row = True
        for row in tsv_reader:
            if is_first_row = True:
                is_first_row = False
                continue
            raw_name = row[1]
            raw_cat_name = row[2]

            ignore_category = False
            for attribute in filter_attributes:
                if attribute in raw_cat_name.lower():
                    ignore_category = True
                    attr_filtering += 1
                    break

            if len(coverage_metadata_dict[raw_name]) < 1:
                threshold_filtering += 1
                ignore_category = True

            if "otherroom" in raw_cat_name.lower():
                raw_cat_name = raw_cat_name.split("/")[0].strip()

            if ignore_category:
                continue
            mapping[raw_name] = raw_cat_name

    print(
        "Post filtering stats - Threshold filtering: {}, Ignore category: {},"
        " Final: {}".format(
            threshold_filtering, attr_filtering, len(mapping.keys())
        )
    )

    return mapping
```

### 2. 修改 `__init__` 方法签名 (行 31)

```python
def __init__(
    self,
    mapping_file: str,
    allowed_categories: Optional[Set[str]] = None,
    coverage_meta_file: Optional[str] = None,  # ← 已经是 Optional，无需修改
    frame_coverage_threshold: Optional[float] = None,
    blacklist_file: Optional[str] = "data/hm3d_meta/blacklist.txt",
) -> None:
    self._mapping = self.limit_mapping(
        self.load_categories(
            mapping_file, coverage_meta_file, frame_coverage_threshold
        ),
        allowed_categories,
        blacklist_file,
    )
```

## 使用方法

### 生成覆盖率元数据
```bash
# 快速测试（1个场景）
python scripts/generate_coverage_meta.py \
    --split train \
    --max-scenes 1 \
    --device-id 0

# 完整生成（所有场景）
python scripts/generate_coverage_meta.py \
    --split train \
    --device-id 0
```

### 生成任务集
```bash
# 使用生成的覆盖率元数据
python ovon/dataset/objectnav_generator.py \
    --split train \
    --scene scene0011 \
    --start-poses-per-object 100 \
    --episodes-per-object 50
```

## 验证

生成后检查文件：
```bash
# 检查文件是否存在
ls -lh data/coverage_meta/train.pkl
ls -lh data/coverage_meta/val.pkl

# 验证内容
python -c "
import pickle
data = pickle.load(open('data/coverage_meta/train.pkl', 'rb'))
print('Categories:', list(data.keys()))
for cat, items in data.items():
    print(f'{cat}: {len(items)} coverage records')
"
```
