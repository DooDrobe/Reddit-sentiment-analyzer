
import streamlit as st
import warnings
warnings.filterwarnings("ignore")
import sys
sys.tracebacklimit = 0
import praw
import requests
import reddit_key as rk #import your unique authentication reddit dev key
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import re
import matplotlib.pyplot as plt
from textblob import TextBlob

st.title('REDDIT SENTIMENT ANALYSER')
def main():
    session = requests.Session()
    session.verify = False 
    reddit = praw.Reddit(
         client_id, 
         client_secret, 
         user_agent,
         requestor_kwargs={'session': session})
    
    
    coin = str()
    coin = str(st.text_input("Insert your coin :",value="bitcoin"))
    #filter = str(st.text_input('filter by(title/body): ',value=""))
    filter = str(st.selectbox('filter by(title/body): ',('title', 'body')))
    limit_scrap = st.number_input('How many post : ',step=1)
    #limit_scrap = int(input('10'))
    search = coin
    posts = []
    ml_subreddit = reddit.subreddit(search)

    for post in ml_subreddit.top(limit=limit_scrap):
    #print(posts)
        posts.append([post.title, post.score, post.id, post.subreddit, post.num_comments, post.selftext, post.created, datetime.fromtimestamp(post.created)])
        #st.write(post)

    posts = pd.DataFrame(posts,columns=['title', 'score', 'id', 'subreddit', 'num_comments', 'body', 'created', 'date'])
    #print(posts)
    df = posts
    
    # st.write("Total Tweets Extracted for Topic '{}' are : {}".format(Topic,len(df.Tweet)))
    # st.write("Total Positive Tweets are : {}".format(len(df[df["Sentiment"]=="Positive"])))
    # st.write("Total Negative Tweets are : {}".format(len(df[df["Sentiment"]=="Negative"])))
    # st.write("Total Neutral Tweets are : {}".format(len(df[df["Sentiment"]=="Neutral"])))

    #DATA CLEANING
    # remove nan value row in body column
    df. dropna(subset = [filter], inplace=True)
    # get only texts
    df = df[[filter]]
    df.columns = ['reddits']
    # Funtion to clean reddits
    def cleanRdt(rdt):
        rdt = re.sub('\\n', '', rdt) # removes the '\n' string
        rdt = re.sub('#[A-Za-z0-9]+', '',rdt) #remove any sting with hashtag
        rdt = re.sub('https?:\/\/\S+', '', rdt) # removes any hyperlinks
        return rdt

    #GET POLARITY
    # function to get subjectivity
    def getSubjectivity(rdt):
        return TextBlob(rdt).sentiment.subjectivity
    # function to get the polarity
    def getPolarity(rdt):
        return TextBlob(rdt).sentiment.polarity

    # function to get the sentiment text
    def getSentiment(score):
        if score < 0:
            return "negative"
        elif score == 0:
            return "neutral"
        else:
            return "positive"

    redsize = len(df)
    st.write ("reddit Size :", redsize)
    
    if len(coin) > 0 :

        #call funtion to clean reddits
        df['cleaned_reddits'] = df['reddits'].apply(cleanRdt)

        # See the Extracted Data : 
        if st.button("See the Extracted Data"):
            #st.markdown(html_temp, unsafe_allow_html=True)
            st.success("Below is the Extracted Data :")
            st.write(df)

        # get the polarity 
        if st.button("Get the polarity"):
            st.success("Analysing polarit")
            st.subheader("Analyzed Cleaned Reddit Polarity")
            # create two new columns called "Subjectivity" & "Polarity"
            df['subjectivity'] = df['cleaned_reddits'].apply(getSubjectivity)
            df['polarity'] = df['cleaned_reddits'].apply(getPolarity)
            df['sentiment'] = df['polarity'].apply(getSentiment)
            st.write(df.head(limit_scrap))

        # get the Scatter plot 
        if st.button("Get the Scatter Polarity"):
            st.set_option('deprecation.showPyplotGlobalUse', False) #ignore warning
            st.success("Analysing polarity")
            st.subheader("Analyzed Cleaned Reddit Scatter Polarity")
            # create two new columns called "Subjectivity" & "Polarity"
            df['subjectivity'] = df['cleaned_reddits'].apply(getSubjectivity)
            df['polarity'] = df['cleaned_reddits'].apply(getPolarity)
            df['sentiment'] = df['polarity'].apply(getSentiment)
            
            plt.figure(figsize=(14,10))

            for i in range(0, redsize):
                plt.scatter(df["polarity"].iloc[[i]].values[0], df["subjectivity"].iloc[[i]].values[0], color="Purple")

            plt.title(search+' Scatter Plot')
            plt.xlabel('polarity')   
            plt.ylabel('subjectivity')
            #plt.savefig(search+' Scatter Plot.png')
            a=plt.show()
            st.pyplot(a)
        
        if st.button("Get the bar chart conclusion"):
            st.set_option('deprecation.showPyplotGlobalUse', False) #ignore warning
            st.success("Analysing polarity")
            st.subheader("Polarity on bar chart")
            # create two new columns called "Subjectivity" & "Polarity"
            df['subjectivity'] = df['cleaned_reddits'].apply(getSubjectivity)
            df['polarity'] = df['cleaned_reddits'].apply(getPolarity)
            df['sentiment'] = df['polarity'].apply(getSentiment)
            
            df['sentiment'].value_counts().plot(kind="bar")
            plt.title(search+" Sentiment Analysis Scatter Plot")
            plt.xlabel("Polarity")
            plt.ylabel("Subjectivity")
            #plt.savefig(search+' Sentiment Analysis Scatter Plot.jpg',dpi=100)
            b = plt.show()
            st.pyplot(b)

    #test field
    

#sidebar
st.sidebar.header("About App")
st.sidebar.info("A reddit scrapper, that scrap your input especially crypto coin/token. Extracted data will be analysed with textblob")
st.sidebar.text("Built with Streamlit")

st.sidebar.header("For Any Queries/Suggestions Please reach out at :")
st.sidebar.info("hatta616@gmail.com")

if __name__ == '__main__':
    main()
