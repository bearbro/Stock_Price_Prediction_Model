import pandas as pd
import os
from py2neo import Graph




out_dir="get_entiry_from_kg"

map_n_p={# 节点类型:属性列表
    "上市公司":["名称","股票代码","曾用名","英文名称","简写名称","公司简介"],
    "非上市公司_股东":["名称"],
    "非上市公司_被控股":["名称"],
    "高管":["名称","简介"]

}
graph = Graph("http://59.78.194.63:37474", auth=("neo4j", "pass"))
graph = Graph("http://brobear.cc:6374", auth=("neo4j", "pass"))


if not os.path.exists(out_dir):
    os.mkdir(out_dir)
    # 查询neo4j节点，获取目标节点数据
    for name,properties in map_n_p.items():
        # "MATCH (n:`上市公司`) RETURN n.`名称`,n.`曾用名`"
        cypher="MATCH (n:`%s`) RETURN id(n),"% name 
        for pi in properties:
            cypher+= "n.`%s`,"% pi
        cypher=cypher[:-1]
        data = graph.run(cypher)
        data=data.to_data_frame()#转化为pandas数据
        data.columns= ["id"]+properties
        file="%s.csv"% name
        data.to_csv(os.path.join(out_dir,file),index=None)

        print(file)
        df=pd.read_csv(os.path.join(out_dir,file),dtype={"股票代码":str,"股票代码2":str,"股东股票代码2":str})
        df.fillna("-", inplace = True)
        df=df.applymap(lambda x: x if x not in["","--","---"] else "-")
        df.to_csv(os.path.join(out_dir,file),index=None)


    
# 合并 各类型节点 到 out_finall_file文件 
out_finall_file="merge.csv"
df_list=[]
for data_path in os.listdir(out_dir):
    if data_path[-4:]!=".csv" or data_path==out_finall_file:
        continue
    df = pd.read_csv(os.path.join(out_dir,data_path), dtype={
                    "股票代码": str, "股票代码2": str, "股东股票代码2": str})
    columns = df.columns.to_list()
    # 统一列名
    for i in range(len(columns)):
        if "简介" in  columns[i]:# xx简介-》简介
            columns[i]="简介"
    df.columns=columns
    df_list.append(df)

df_final=df_list[0]
for df_i in df_list[1:]:
    df_final=pd.concat([df_final,df_i],axis=0)
df_final=df_final.fillna("-")
df_final.to_csv(os.path.join(out_dir,out_finall_file),index=None)