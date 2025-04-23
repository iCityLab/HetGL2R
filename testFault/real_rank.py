# -*- coding: utf-8 -*-
# Create Time  :  2023/12/19 14:05
# Author       :  xjr17
# File Name    :  real_rank.PY
# software     :  PyCharm
import bs4
import pandas as pd
import xlsxwriter
import lxml.etree as ET
from bs4 import BeautifulSoup

# 读取xml文件，写入excel
def xmlToExcelFcd(file_excel):
    # 创建一个excel文件，并添加一个sheet，命名为orders
    workbook = xlsxwriter.Workbook(file_excel)
    sheet = workbook.add_worksheet('orders')

    # 设置粗体
    bold = workbook.add_format({'bold': True})

    # 先在第一行写标题，用粗体
    sheet.write('A1', u'link', bold)
    sheet.write('B1', u'time', bold)
    sheet.write('C1', u'eff_link', bold)
    sheet.write('D1', u'speed', bold)
    # 从第二行开始录入数据
    row = 2

    for i in range(1,111):
        path = r'D:\Project\PythonProject\HetGL2R\RN\fcd'+str(i)+'.xml'
        file_xml = path
        # 打开xml文件，并以此创建一个bs对象
        xml = open(file_xml, 'rb').read()  # 注意使用二进制模式读取
        doc = BeautifulSoup(xml, 'lxml-xml')
        print(i)
        #xml = open(file_xml, 'r')
        #doc = bs4.BeautifulSoup(xml, 'html.parser')
        # 筛选出所有的<timestep>，这里使用的是CSS选择器
        time = doc.select('timestep')
        for t in time:
            ti = t.attrs["time"]
            if float(ti) <= 200:
                vehicle = t.select('vehicle')
                for v in vehicle:
                    if float(v.attrs["speed"]) <= 1.389:
                        veff = v.attrs["lane"]
                        speed = v.attrs["speed"]
                        sheet.write('A%d' % row, 'link'+str(i))
                        sheet.write('B%d' % row,ti)
                        sheet.write('C%d' % row, veff)
                        sheet.write('D%d' % row,speed)
                        row += 1
            # 关闭文件
            #xml.close()
    workbook.close()


def real_score(table,r):
    table = pd.read_table(table, names=['link', 'time','eff_link','speed'], sep='\t')
    #print(f"表格的行数是: {table.shape[0]}")
    # 去重操作
    #table = table.drop_duplicates(subset=['link', 'time', 'eff_link', 'speed'])
    #print(f"表格的行数是: {table.shape[0]}")
    real_out = {}
    real_rank = {}
    link_name = []
    time_list = list(range(1, 201))
    with open(r'D:\Project\PythonProject\HetGL2R\testFault\segmentsIDrn.txt') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            link_name.append(line.lower())

    #print(len(table))
    for name in link_name:
        real_in = dict([(key,0) for key in time_list])
        for i in table.index:
            if table['link'][i] == name:
                if table['speed'][i] <= 1.389:#1.389
                    real_in[table['time'][i]] += (1+table['speed'][i])

        real_out[name] = real_in
        #print(real_in)
    for key1 in real_out.keys():
        eff = 0
        for key2 in real_out[key1].keys():
            eff += (r**key2) * real_out[key1][key2]
        real_rank[key1] = eff

    return real_rank



if __name__ == '__main__':
    file = r'D:\Project\PythonProject\HetGL2R\testFault\rn.xlsx'
    #xmlToExcelFcd(file)
    eff = r'D:\Project\PythonProject\HetGL2R\testFault\rn.txt'
    real_rank = real_score(eff,0.9)
    print(real_rank)
    order = sorted(real_rank.items(), key=lambda x: x[1], reverse=True)
    print(order)
    with open(r'D:\Project\PythonProject\HetGL2R\testFault\real_rn.txt', 'w') as f:
        for i in order:
            f.write(str(i)+'\n')
        f.close()


