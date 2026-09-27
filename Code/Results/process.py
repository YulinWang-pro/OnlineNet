import os
import re
import pandas as pd
import argparse
from pathlib import Path


def aggregate_results(base_dir='.', output_dir='aggregated_results'):
    """
    遍历目录，在内存中聚合所有符合条件的文件，并打印详细的处理流程日志。
    目录结构：Base_Dir -> Folder_A -> Model_Type_Folder -> 结果.csv
    """
    base_path = Path(base_dir).resolve()
    output_path = Path(output_dir).resolve()

    # 正则表达式：匹配 '{数据集}_prob_{参数}final.csv'
    pattern = re.compile(r"^(.+)_prob_(.+)final\.csv$")
    all_data = []

    print("🚀 开始执行数据聚合任务...")
    print(f"📂 目标扫描目录: {base_path}")
    print(f"📁 结果输出目录: {output_path}")
    print("-" * 60)

    processed_count = 0
    current_folder_a = None

    # 使用 os.walk 遍历提取所有基础数据
    for root, dirs, files in os.walk(base_path):
        current_dir = Path(root).resolve()

        # 排除存放结果的输出目录，防止无限循环或重复读取
        if output_path in current_dir.parents or current_dir == output_path:
            continue

        # 预先筛选出当前文件夹下符合命名规则的文件
        matching_files = [f for f in files if pattern.match(f)]

        if not matching_files:
            continue  # 如果当前目录没有符合条件的文件，直接跳过

        # 提取层级目录名称
        model_type = current_dir.name
        folder_a = current_dir.parent.name

        # 打印：进入新的外层文件夹 (Folder_A)
        if folder_a != current_folder_a:
            print(f"\n📦 进入外层文件夹 (Folder_A): [{folder_a}]")
            current_folder_a = folder_a

        # 打印：当前正在处理的模型文件夹 (Model_Type)
        print(f"   🛠️  扫描模型文件夹 (Model_Type): [{model_type}] - 发现 {len(matching_files)} 个实验结果")

        for file in matching_files:
            match = pattern.match(file)
            dataset_name = match.group(1)
            parameter = match.group(2)

            file_path = current_dir / file
            try:
                df = pd.read_csv(file_path)

                # 修复第一列的无表头问题
                if df.columns[0].startswith('Unnamed'):
                    df.rename(columns={df.columns[0]: 'Config/Hyperparams'}, inplace=True)

                # 插入标识列
                df.insert(0, 'Parameter', parameter)
                df.insert(0, 'Dataset', dataset_name)
                df.insert(0, 'Model_Type', model_type)
                df.insert(0, 'Folder_A', folder_a)

                all_data.append(df)
                processed_count += 1

                # 打印：单个文件的处理情况
                print(f"       📄 成功提取: {file}  (数据集: {dataset_name}, 参数: {parameter})")

            except Exception as e:
                print(f"       ❌ 读取文件失败 {file_path}: {e}")

    print("\n" + "-" * 60)
    if not all_data:
        print("⚠️ 未找到任何符合 '数据集_prob_参数final.csv' 格式的文件，程序结束。")
        return

    print(f"✅ 所有文件读取完毕！共成功处理了 {processed_count} 个结果文件。")
    print("⏳ 正在内存中拼接数据，并进行全局多级排序 (Dataset -> Model_Type -> Parameter -> Folder_A)...")

    # 直接在内存中合并成最终的 1 个 DataFrame
    final_merged_df = pd.concat(all_data, ignore_index=True)

    # 多级排序
    final_merged_df.sort_values(by=['Dataset', 'Model_Type', 'Parameter', 'Folder_A'], inplace=True)

    # 创建输出目录
    output_path.mkdir(parents=True, exist_ok=True)

    # 导出唯一一个最终文件
    final_output_file = output_path / "Final_All_Aggregated.csv"
    final_merged_df.to_csv(final_output_file, index=False)

    print(f"🎉 终极聚合完成！(共聚合数据 {len(final_merged_df)} 行)")
    print(f"💾 最终文件已保存至: {final_output_file}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="聚合各文件夹下的实验结果为单个 CSV，并打印处理流程")
    parser.add_argument('--dir', type=str, default='.',
                        help="目标扫描目录，默认扫描当前目录('.')")
    parser.add_argument('--out', type=str, default='aggregated_results',
                        help="输出结果的新文件夹名称，默认为 'aggregated_results'")

    args = parser.parse_args()
    aggregate_results(args.dir, args.out)