import pandas as pd
import streamlit as sl

sl.title("Stocks prediction")
sl.header("Ever wondered whether to buy a stock or not !")
sl.subheader(
    "Look at the Stocks to see how they perform and guess what you have an option to play with this wonderful model by yourself"
    )
sl.info("As of now this app runs only with HDFC  and GoldmanSachs stocks and further devlopements are being made")   
sl.image('stock1.jpg',use_column_width=True)   
Stocklist=['HDFC','GoldmanSachs','Apple','Amazon','Bitcoin']

if(sl.checkbox("Select Stock")):
    stockChosen=sl.multiselect('Select a stock',Stocklist)
    if (stockChosen==[]):
        sl.error("Warning : Please choose a Stock")    
    
    elif(len(stockChosen)>=2):
        sl.error("Warning :Please Choose Only one at a time")
    elif(stockChosen[0]==Stocklist[2]):
        sl.error("Oops :( We can see your interest in this Stock but it is unfortunately still Under development")
    elif(stockChosen[0]==Stocklist[3]):
        sl.error("Oops :( We can see your interest in this Stock but it is unfortunately still Under development")        
    elif(stockChosen[0]==Stocklist[4]):
        sl.error("Oops :( We can see your interest in this Stock but it is unfortunately still Under development")

    elif(stockChosen[0]==Stocklist[0]):
        df=pd.read_csv('HDFCBANK.csv')
        sl.warning("NOTE : Stocks tend to fluctuate quickly so kindly invest in them at your own risk")
        sl.info("By clicking the checkbox below you can view the data by which the model has been trained")
        if(sl.checkbox("Show Data")):
            sl.write("Below are the data of stocks from the day the company went public")
            sl.dataframe(df)
        if(sl.checkbox("Meaningful data representation")):
            sl.markdown("**Insights** of the data ")
            basicInfo=df.describe()
            sl.dataframe(basicInfo)
            sl.markdown("*Bar graph*")
            sl.bar_chart(basicInfo)
        
        sl.video('stockvideo.mp4') 
        from sklearn.ensemble import RandomForestRegressor
        X=df.iloc[:, 0].values
        PreVal=X.shape[0]
        X=X.reshape(PreVal,1)
        Y=df.iloc[:, [1,4]].values
        from sklearn.model_selection import train_test_split
        X_train,X_test,Y_train,Y_test=train_test_split(X,Y)
        model=RandomForestRegressor(n_estimators=19)
        model.fit(X_train,Y_train)
        pred=model.predict(X_test)
        sl.markdown('**Sample Predictions**')
        sl.line_chart(pred[0:150])
        Day=1
        sl.markdown("**Play** with the model yourself to **Predict**")
        sl.markdown(' With the **Slider** given below choose which day from today you want to predict ')
        sl.slider('Day',min_value=1,max_value=30)
        predicted=model.predict([[Day+PreVal]])
        sl.write("The Predicted opening price and closing price of Stocks are ",predicted)
                   
    elif(stockChosen[0]==Stocklist[1]):
        df=pd.read_csv('GS.csv')
        sl.warning("NOTE : Stocks tend to fluctuate quickly so kindly invest in them at your own risk")
        sl.info("By clicking the checkbox below you can view the data by which the model has been trained")
        if(sl.checkbox("Show Data")):
            sl.write("Below are the data of stocks from the day the company went public")
            sl.dataframe(df)
        if(sl.checkbox("Meaningful data representation")):
            sl.markdown("**Insights** of the data ")
            basicInfo=df.describe()
            sl.dataframe(basicInfo)
            sl.markdown("*Bar graph*")
            sl.bar_chart(basicInfo)
        sl.video('Analysis.mp4') 
        from sklearn.ensemble import RandomForestRegressor
        X=df.iloc[:, 0].values
        X=X.reshape(5378,1)
        Y=df.iloc[:, [1,4]].values
        from sklearn.model_selection import train_test_split
        X_train,X_test,Y_train,Y_test=train_test_split(X,Y)
        model=RandomForestRegressor(n_estimators=19)
        model.fit(X_train,Y_train)
        pred=model.predict(X_test)
        sl.markdown('**Sample Predictions**')
        sl.line_chart(pred[0:150])
        Day=1
        PreVal=X.shape[0]
        sl.markdown("**Play** with the model yourself to **Predict**")
        sl.markdown(' With the **Slider** given below choose which day from today you want to predict ')
        sl.slider('Day',min_value=1,max_value=30)
        predicted=model.predict([[Day+PreVal]])
        sl.write("The Predicted opening price and closing price of Stocks are ",predicted)

if(sl.button("About this app")):
    sl.info("This webapp is still under construction and works with limited number of data and some basic machine learning model behind it ")
if(sl.button("Version")):
    sl.info("Version:0.3")  
myRate=sl.sidebar.slider("Rate",max_value=5,min_value=1)
if(sl.checkbox("Contact User(Currently unavailable)")):
    t1=sl.text_input("Name")
    t2=sl.text_input("Email-Id")
    t3=sl.text_area("Enter your Message here ")
    if(t3 != ''):
        sl.write("Thank you %s ,We will contact you soon"%t1)
    f=open('message.txt','a+')
    f.write("-----------------------\n")
    f.write("Name : %s\n"%t1)
    f.write("Name : %s\n"%t2)
    f.writelines("The Message :  %s\n" %t3)
    f.write("The rate is %s\n"%myRate)
    f.close()
    




    




    






