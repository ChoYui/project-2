#2-1
import pandas as pd
import pandas_datareader.data as reader

data_df = pd.read_csv('2019_kbo_for_kaggle_v2.csv')


print('1)')
def top_players(dataset_df,x):
    new_ds = dataset_df[dataset_df['p_year']==(x+1)]
    r_h = new_ds.sort_values(by='H',ascending=False)
    r_avg = new_ds.sort_values(by='avg',ascending=False)
    r_HR = new_ds.sort_values(by='HR',ascending=False)
    r_OBP = new_ds.sort_values(by='OBP',ascending=False)
    print('Top 10 hits:\n',r_h.head(10)['batter_name'])
    print('\nTop 10 batting average:\n',r_avg.head(10)['batter_name'])
    print('\nTop 10 homerun:\n',r_HR.head(10)['batter_name'])
    print('\nTop 10 on-base per:\n',r_OBP.head(10)['batter_name'])

for i in range(2015,2019):
    print("\nIn ",i,'\n')
    top_players(data_df,i)


print('\n2)')
data_filtered = data_df[data_df['p_year']==2019]
position_list = ['포수', '1루수', '2루수', '3루수', '유격수', '좌익수', '중견수', '우익수']
for i in position_list:
  new_data = data_filtered[data_filtered['cp']==i][['batter_name','war']]
  war_max = new_data['war'].max()
  result = new_data[new_data['war']== war_max]['batter_name']
  print('The highest war by',i,'의 index와 선수이름: ',result,'\n')


print('\n3)')
data_list = ['R','H','HR','RBI','SB','war','avg','OBP','SLG','salary']
new3_df = data_df[data_list]
#print(new3_df)

c=-2
a=''

for i in data_list[:-1]:
  corr_data = new3_df[i].corr(new3_df['salary'])
  if c<corr_data:
    c = corr_data
    a = i

print('The highest correlation with salary: ',a)

