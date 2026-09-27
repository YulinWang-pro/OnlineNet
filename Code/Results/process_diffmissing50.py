import pandas as pd
import argparse
from pathlib import Path


def create_difmissing25(input_file, output_dir='aggregated_results'):
    """
    读取全局汇总数据，生成 difmissing25 结果表。
    过滤条件: Parameter == 50
    格式: [Dataset, metric, Model_Type_1, Model_Type_2, ...]
    """
    input_path = Path(input_file).resolve()
    output_path = Path(output_dir).resolve()

    if not input_path.exists():
        print(f"❌ 找不到输入文件: {input_path}")
        return

    print(f"📥 正在读取数据: {input_path}")
    df = pd.read_csv(input_path)

    # 1. 过滤 Parameter == 25 的行
    # 注意：为了防止类型不匹配（有的是数字25，有的是字符串'25'），统一转成字符串比较
    df_25 = df[df['Parameter'].astype(str) == '50'].copy()

    if df_25.empty:
        print("⚠️ 数据中没有找到 Parameter 为 50 的行。")
        return

    print("⏳ 正在进行数据转换 (Melt & Pivot) ...")

    # 定义需要被转换的指标列
    metrics = ['Num. Errors', 'Accuracy', 'AUROC', 'AUPRC', 'Bal. Accuracy', 'Time']

    # 检查这些指标列是否都存在于数据中
    available_metrics = [m for m in metrics if m in df_25.columns]
    if len(available_metrics) < len(metrics):
        print(f"⚠️ 警告: 数据中缺失部分指标列，将仅处理存在的列: {available_metrics}")

    # 2. 将宽表融化（Melt）成长表
    # 把所有 metric 对应的列名压缩到一列名为 'metric' 的列中，对应的值压缩到 'value' 列中
    melted_df = pd.melt(
        df_25,
        id_vars=['Dataset', 'Model_Type'],
        value_vars=available_metrics,
        var_name='metric',
        value_name='value'
    )

    # 3. 将长表透视（Pivot）成目标宽表
    # 将 Model_Type 展开为表头
    pivot_df = melted_df.pivot_table(
        index=['Dataset', 'metric'],
        columns='Model_Type',
        values='value',
        aggfunc='first'
    ).reset_index()

    # 清除透视产生的列名层级
    pivot_df.columns.name = None

    # 4. 排序美化
    # 为了保证 metric 列严格按照你要求的顺序排列，我们将其设置为分类变量(Categorical)
    pivot_df['metric'] = pd.Categorical(pivot_df['metric'], categories=available_metrics, ordered=True)
    pivot_df.sort_values(by=['Dataset', 'metric'], inplace=True)

    # 创建输出目录并保存结果
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / "difmissing50.csv"

    pivot_df.to_csv(output_file, index=False)

    print(f"✅ 生成成功！difmissing50 表已保存至: {output_file}")

    # 打印前几行预览
    print("\n📊 结果预览:")
    print(pivot_df.head(10).to_string())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="生成 Parameter=50 的各项指标对比表")
    parser.add_argument('--input', type=str, default='aggregated_results/Final_All_Aggregated.csv',
                        help="输入的全局汇总CSV文件路径")
    parser.add_argument('--out', type=str, default='aggregated_results',
                        help="输出结果的文件夹")

    args = parser.parse_args()
    create_difmissing25(args.input, args.out)