import pandas as pd
from multiprocessing import Pool,cpu_count
import pandas as pd
import tushare as ts
from Tool.toolkit import describe_group


pro = ts.pro_api('92bfb7a5df70a386927cf4cb1c2b5809df1f45ba402f32dc605111f6')


#获取申万三级级行业列表
df2 = pro.index_classify(level='L2', src='SW2021')
df=pd.read_feather('D:/project/quant_trade/data.feather')

for row in df2.iterrows():
    a=row[1]
    b=a['index_code']
    c=a['industry_name']
    sw=pro.index_member(index_code=b,is_new='Y')
    if sw.empty==True:
        continue
    data=df[df['ts_code'].isin(sw['con_code'].str[:-3])]
    grouped = data.groupby('trade_date')
    if __name__ == '__main__':
        with Pool(processes=cpu_count() - 2) as pool:
             results=pool.map(describe_group, [group for _, group in grouped])
    for  trade_date, group_df in zip(grouped.groups.keys(), results):
        group_df['trade_date'] = trade_date
    combined_df = pd.concat(results)
    write=pd.ExcelWriter(path='D:/project/quant_trade/industry/'+c+'.xlsx')
    combined_df.to_excel(excel_writer=write,sheet_name='dec',index=None)
    data.to_excel(excel_writer=write,sheet_name='data',index=None)
    write.save()
    
    
    
    
    
    
    
    
    
    
    
    