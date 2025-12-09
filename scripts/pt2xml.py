import xml.etree.ElementTree as ET
from xml.dom import minidom

def pt_to_xml_deinterleaving(result, save_file_path):
    root = ET.Element('Table')
    header_row = ET.SubElement(root, 'Row')
    headers = ['LISTNUM', 'LABEL']
    for header in headers:
        cell = ET.SubElement(header_row, 'Cell')
        data = ET.SubElement(cell, 'Data')
        data.text = header

    for i, label in enumerate(result):
        data_row = ET.SubElement(root, 'Row')

        listnum_cell = ET.SubElement(data_row, 'Cell')
        listnum_data = ET.SubElement(listnum_cell, 'Data')
        listnum_data.text = str(i+1)

        label_cell = ET.SubElement(data_row, 'Cell')
        label_data = ET.SubElement(label_cell, 'Data')
        label_data.text = f"{label:04d}"

    rough_string = ET.tostring(root, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    pretty_xml = reparsed.toprettyxml(indent="    ")

    with open(save_file_path, 'w') as xml_file:
        xml_file.write(pretty_xml)