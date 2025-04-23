# -*- coding: utf-8 -*-
# Create Time  :  2024/8/19 16:34
# Author       :  xjr17
# File Name    :  modify_speed.PY
# software     :  PyCharm

import subprocess
import time

import xml.etree.ElementTree as ET

# 打开并读取文件内容
with open('segmentsIDrn.txt', 'r') as file:
    # 读取所有行，并去除每行末尾的换行符
    ids = [line.strip() for line in file.readlines()]

# # 打印数组内容
# print(ids)

# 读取XML文件

counter = 0

# 遍历所有的edge元素
for edge_id in ids:
    tree = ET.parse(r'D:\Project\PythonProject\HetGL2R\RN\rn.net.xml')
    root = tree.getroot()
    target_edges = [edge for edge in root.findall('edge') if edge.get('id').replace(":", "") == edge_id]
    #edge_id = edge.get('id').replace(":", "")
    for edge in target_edges:
        counter += 1
        # 遍历edge中的所有lane元素
        for lane in edge.findall('lane'):
            # 获取当前的speed值并将其乘以0.1
            current_speed = float(lane.get('speed'))
            new_speed = current_speed * 0.1
            # 将新的speed值更新到lane元素中
            lane.set('speed', f'{new_speed:.2f}')
        # 将修改后的XML保存到文件
        tree.write(r'D:\Project\PythonProject\HetGL2R\RN\rn_modified.net.xml', encoding='UTF-8',
                       xml_declaration=True)
        print(edge_id)
        # 读取XML文件
        tree1 = ET.parse(r'D:\Project\PythonProject\HetGL2R\RN\rn.sumocfg')
        root = tree1.getroot()

        # 修改tripinfo-output和fcd-output的文件名
        for output in root.findall('.//output'):
            tripinfo_output = output.find('.//tripinfo-output')
            if tripinfo_output is not None:
                tripinfo_output.set('value', f'tripinfos{counter}.xml')

            fcd_output = output.find('.//fcd-output')
            if fcd_output is not None:
                fcd_output.set('value', f'fcd{counter}.xml')

        # 将修改后的XML保存回文件
        tree1.write(r'D:\Project\PythonProject\HetGL2R\RN\rn.sumocfg', encoding='UTF-8',
                   xml_declaration=True)

        print(f"Updated map.sumocfg with tripinfos{counter}.xml and fcd{counter}.xml")

        sumo_cfg_file = r'D:\Project\PythonProject\HetGL2R\RN\rn.sumocfg'
        sumo_command = r'D:\Study\SUMO\sumo-1.18.0\bin\sumo-gui'
        command = [sumo_command, '-c', sumo_cfg_file]
        # 启动SUMO
        try:
            process = subprocess.run(command)
            #time.sleep(1800)
            #process.terminate()
            #subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"SUMO仿真失败: {e}")
        else:
            print("SUMO仿真成功完成。")

        # 将修改后的XML保存到文件
#tree.write('D:\Project\PythonProject\experiment_one\Test\SY_OD_City\map_modified.net.xml', encoding='UTF-8', xml_declaration=True)

