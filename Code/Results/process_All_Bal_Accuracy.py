import pandas as pd
import argparse
from pathlib import Path


def create_bal_accuracy_pivot(input_file, output_dir='aggregated_results'):
    """
    读取全局汇总数据，生成 All_Bal_Accuracy 结果表。
    格式: [Dataset, Parameter, Model_Type_1, Model_Type_2, ...]
    内容: Bal. Accuracy 的值
    """
    input_path = Path(input_file).resolve()
    output_path = Path(output_dir).resolve()

    if not input_path.exists():
        print(f"❌ 找不到输入文件: {input_path}")
        return

    print(f"📥 正在读取数据: {input_path}")
    df = pd.read_csv(input_path)

    # 检查是否存在需要的列
    required_columns = ['Dataset', 'Parameter', 'Model_Type', 'Bal. Accuracy']
    for col in required_columns:
        if col not in df.columns:
            print(f"❌ 数据中缺少必要的列: {col}")
            return

    print("⏳ 正在生成基于 Bal. Accuracy 的透视对比表...")

    # 使用 pivot_table 生成透视表
    # index: 保持在左侧的行 (Dataset, Parameter)
    # columns: 需要展开成表头的列 (Model_Type)
    # values: 填入单元格的具体数值 (Bal. Accuracy)
    # aggfunc='first': 因为值是类似 '60.38(0.62)' 的字符串，我们直接取第一个匹配到的字符串即可
    pivot_df = df.pivot_table(
        index=['Dataset', 'Parameter'],
        columns='Model_Type',
        values='Bal. Accuracy',
        aggfunc='first'
    ).reset_index()

    # 清除透视产生的列名层级（让表头看起来更干净）
    pivot_df.columns.name = None

    # 为了保证 Parameter = [25, 50, 75] 能按照数字大小正确排序，我们将其临时转为数值类型排序
    try:
        pivot_df['Parameter_Num'] = pd.to_numeric(pivot_df['Parameter'])
        pivot_df.sort_values(by=['Dataset', 'Parameter_Num'], inplace=True)
        pivot_df.drop(columns=['Parameter_Num'], inplace=True)
    except:
        # 如果 Parameter 中含有无法转为数字的字符，则按照默认字符串排序
        pivot_df.sort_values(by=['Dataset', 'Parameter'], inplace=True)

    # 创建输出目录并保存结果
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / "All_Bal_Accuracy.csv"

    pivot_df.to_csv(output_file, index=False)

    print(f"✅ 生成成功！直观对比表已保存至: {output_file}")

    # 在终端简单打印一下前几行预览
    print("\n📊 结果预览:")
    print(pivot_df.head(6).to_string())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="生成基于 Bal. Accuracy 的模型对比表")
    parser.add_argument('--input', type=str, default='aggregated_results/Final_All_Aggregated.csv',
                        help="输入的全局汇总CSV文件路径")
    parser.add_argument('--out', type=str, default='aggregated_results',
                        help="输出结果的文件夹")

    args = parser.parse_args()
    create_bal_accuracy_pivot(args.input, args.out)