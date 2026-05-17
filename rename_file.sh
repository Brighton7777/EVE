#!/bin/bash

# 目标目录
TARGET_DIR="./all_results/chair"

# 检查目录是否存在
if [ ! -d "$TARGET_DIR" ]; then
    echo "错误: 目录 $TARGET_DIR 不存在"
    exit 1
fi

echo "开始处理目录: $TARGET_DIR"
echo "将文件名中的 jsd_1、jsd_2 替换为 eve，并删除 opera"
echo "----------------------------------------"

# 查找所有包含 jsd_1、jsd_2 或 opera 的文件
find "$TARGET_DIR" -type f \( -name "*jsd_1*" -o -name "*jsd_2*" -o -name "*opera*" \) | while read -r file; do
    # 获取文件所在目录和文件名
    dir=$(dirname "$file")
    filename=$(basename "$file")
    
    # 替换文件名中的 jsd_1 和 jsd_2 为 eve，删除 opera
    new_filename=$(echo "$filename" | sed 's/jsd_1/eve/g; s/jsd_2/eve/g; s/opera//g')
    
    # 构建新文件路径
    new_file="$dir/$new_filename"
    
    # 如果新文件名与原文件名不同，则重命名
    if [ "$file" != "$new_file" ]; then
        mv "$file" "$new_file"
        echo "重命名: $filename -> $new_filename"
    fi
done

echo "----------------------------------------"
echo "处理完成!"