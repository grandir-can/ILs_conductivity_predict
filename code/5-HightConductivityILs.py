import pandas as pd
pd.set_option('display.max_columns', None) # 显示完整的列
pd.set_option('display.max_rows', None) # 显示完整的行
pd.set_option('display.expand_frame_repr', False) # 设置不折叠数据
pd.set_option('display.max_colwidth', 100)

df = pd.read_csv('../newILs/new_ILs_conductivity.csv')

l = int(len(df)*0.05)

df_275 = df.sort_values(by='predicted ionic conductivity(275K)',ascending=False)
df_mer = df_275[:l].loc[:,['cation','anion','predicted ionic conductivity(275K)']]
tem_key = ['predicted ionic conductivity(290K)','predicted ionic conductivity(300K)','predicted ionic conductivity(310K)','predicted ionic conductivity(325K)']
for t in tem_key:
    df_t =  df.sort_values(by=t,ascending=False)
    df_t_top5 = df_t[:l].loc[:, ['cation', 'anion', t]]
    df_mer = pd.merge(df_mer, df_t_top5, on=['cation', 'anion'])

# cation = df_mer['cation'].values
# anion = df_mer['anion'].values
#
# cation = set(cation)
# anion = set(anion)
#
# print((len(cation)))
# print(len(anion))

df_mer.to_csv('../newILs/hight_conductivity_ILs.csv',index =False)



