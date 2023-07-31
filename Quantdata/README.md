# Quantdata

fuction：
catchstd      	用trains的数据训练出预测值和真实值的误差作为标准差生成每股std文件
catchsignal     用std文件以及tushare数据生成卖出信号并生成个股回测数据
core            用多线程调用catchsignal生成多股回测数据
Getdata         tushare调用股票数据