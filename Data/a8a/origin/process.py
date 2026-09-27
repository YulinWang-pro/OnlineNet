def merge_libsvm_files(file1, file2, output_file):
    with open(output_file, 'wb') as outfile:
        # 读取第一个文件
        with open(file1, 'rb') as f1:
            content = f1.read()
            outfile.write(content)
            # 如果文件末尾没有换行符，补一个换行符
            if content and not content.endswith(b'\n'):
                outfile.write(b'\n')

        # 读取第二个文件并追加
        with open(file2, 'rb') as f2:
            outfile.write(f2.read())


# 执行合并
merge_libsvm_files('a8a', 'a8a.t', 'a8a.txt')
print("合并完成，新文件名为: a8a")