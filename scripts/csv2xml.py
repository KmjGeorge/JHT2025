import xml.etree.ElementTree as ET
from xml.dom import minidom


def txt_to_xml_deinterleaving(txt_file_path, save_file_path):
            # 创建XML根元素
            root = ET.Element("Table")

            # 添加表头行
            header_row = ET.SubElement(root, "Row")
            headers = ["LISTNUM", "LABEL", "TOAdot", "Freq", "PWdot", "SNR"]
            for header in headers:
                cell = ET.SubElement(header_row, "Cell")
                data = ET.SubElement(cell, "Data")
                data.text = header

            # 读取TXT文件并处理数据
            with open(txt_file_path, 'r') as file:
                lines = file.readlines()
                lines = lines[1:]
                # 假设每行数据以逗号分隔，且顺序与表头一致
                for i, line in enumerate(lines):
                    # 跳过空行
                    if not line.strip():
                        continue
                    # 分割数据行
                    values = line.strip().split('\t')

                    # 创建数据行
                    data_row = ET.SubElement(root, "Row")

                    # 添加LISTNUM（行号）
                    listnum_cell = ET.SubElement(data_row, "Cell")
                    listnum_data = ET.SubElement(listnum_cell, "Data")
                    listnum_data.text = str(i+1)

                    # 添加其他数据
                    for j in range(len(values)):
                        cell = ET.SubElement(data_row, "Cell")
                        data = ET.SubElement(cell, "Data")
                        # 转化标签为4位格式，从0001开始
                        if j == 0:
                            values[j] = int(float(values[j])) + 1
                            values[j] = f"{values[j]:04d}"
                        data.text = values[j]

            # 生成XML树
            # tree = ET.ElementTree(root)

            # 美化XML输出
            rough_string = ET.tostring(root, 'utf-8')
            reparsed = minidom.parseString(rough_string)
            pretty_xml = reparsed.toprettyxml(indent="    ")

            # 写入XML文件
            with open(save_file_path, 'w') as xml_file:
                xml_file.write(pretty_xml)


def txt_to_xml_recoginition(txt_label1_path, txt_label2_path, save_file_path):
    """
    从TXT文件读取数据并生成新的XML格式

    参数:
        txt_file_path: 输入的TXT文件路径
        xml_file_path: 输出的XML文件路径
    """
    # 创建XML根元素
    root = ET.Element("Table")

    # 添加表头行
    header_row = ET.SubElement(root, "Row")

    # 添加型号LABEL表头
    model_cell = ET.SubElement(header_row, "Cell")
    model_data = ET.SubElement(model_cell, "Data")
    model_data.text = "型号LABEL"

    # 添加个体LABEL表头
    individual_cell = ET.SubElement(header_row, "Cell")
    individual_data = ET.SubElement(individual_cell, "Data")
    individual_data.text = "个体LABEL"

    # 读取TXT文件并处理数据
    with open(txt_label1_path, 'r', encoding='utf-8') as file:
        _ = file.readline()
        data1 = file.readline()
        _, label1 = data1.strip().split('\t')
        label1 = int(float(label1))
        if label1 == 99:    # 未知
            label1 = 0

    with open(txt_label2_path, 'r', encoding='utf-8') as file:
        _ = file.readline()
        data1 = file.readline()
        _, label2 = data1.strip().split('\t')
        label2 = int(float(label2))

    data_row = ET.SubElement(root, "Row")

    # 添加型号LABEL
    model_cell = ET.SubElement(data_row, "Cell")
    model_data = ET.SubElement(model_cell, "Data")
    model_data.text = f"{label1:04d}"

    # 添加个体LABEL (格式化为4位)
    individual_cell = ET.SubElement(data_row, "Cell")
    individual_data = ET.SubElement(individual_cell, "Data")
    individual_data.text = f"{label2:04d}"

    # # 生成XML树
    # tree = ET.ElementTree(root)

    # 美化XML输出
    rough_string = ET.tostring(root, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    pretty_xml = reparsed.toprettyxml(indent="    ")

    # 写入XML文件
    with open(save_file_path, 'w', encoding='utf-8') as xml_file:
        xml_file.write(pretty_xml)

# 使用示例
if __name__ == "__main__":
    txt_to_xml_deinterleaving(r"G:\datasets\2025金海豚初赛数据\提交格式\result_fx.txt",
                              r"G:\datasets\2025金海豚初赛数据\提交格式\output_fx.xml")
    txt_to_xml_recoginition(r"G:\datasets\2025金海豚初赛数据\提交格式\result_sb_1.txt", r"G:\datasets\2025金海豚初赛数据\提交格式\result_sb_2.txt",
                              r"G:\datasets\2025金海豚初赛数据\提交格式\output_sb.xml")