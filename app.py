from praw.reddit import Submission
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
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

st.title('REDDIT SENTIMENT ANALYZER')
def main():
    session = requests.Session()
    session.verify = False 
    reddit = praw.Reddit(
        #  client_id=rk.client_id, 
        #  client_secret=rk.client_secret, 
        #  user_agent=rk.user_agent,
        client_id=st.secrets["client_id"], 
        client_secret=st.secrets["client_secret"], 
        user_agent=st.secrets["user_agent"],
        requestor_kwargs={'session': session})
    
    #COLUMN MAKER
    col1, col2= st.columns([1.5,4])
    #COLUMN 1
    with col1:
        coin = str()
        coin = str(st.text_input("Insert your coin :",value="bitcoin"))
        limit_scrap = st.number_input('How many post (max : 1000) : ',step=1)
        search = coin
        posts = []
        submission = reddit.subreddit(search)

        #Select Box topic
        New = submission.new(limit=limit_scrap)
        Hot = submission.hot(limit=limit_scrap)
        Top = submission.top(limit=limit_scrap)
        Controversial = submission.controversial(limit=limit_scrap)
        sort_by = str(st.selectbox('Choose Topic: ',('New', 'Top', 'Hot', 'Controversial' )))        
        x=eval(sort_by)
        
        #for post in submission.x(limit=limit_scrap):
        for post in x:
            posts.append([post.title, post.score, post.id, post.subreddit, post.num_comments, post.selftext, post.created, datetime.fromtimestamp(post.created), post.url])
            
        posts = pd.DataFrame(posts,columns=['title', 'score', 'id', 'subreddit', 'num_comments', 'body', 'created', 'date', 'url'])      
        
        df = posts

        #Data raw
        raw = df[['title', 'score', 'id',  'num_comments', 'body', 'date', 'url']]
        raw.columns = ['title', 'score', 'id', 'num_comments', 'body', 'date', 'url']   
              
        #DATA CLEANING
        # remove nan value row in body column
        df. dropna(subset = ["body"], inplace=True)

        # combine title and body column
        df['a'] = df["title"].astype(str)
        df['b'] = df["body"].astype(str)
        df = df['b'] + df['a']
        #convert data series to data_frame
        df = df.to_frame('reddits')
        
        
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
        def prepCloud(coin_text,coin):
            coin = str(coin).lower()
            coin=' '.join(re.sub('([^0-9A-Za-z \t])', ' ', coin).split())
            coin = re.split("\s+",str(coin))
            stopwords = set(STOPWORDS)
            stopwords.update(coin) ### Add our coin in Stopwords, so it doesnt appear in wordClous
            ###
            text_new = " ".join([txt for txt in coin_text.split() if txt not in stopwords])
            return text_new
        
        #call funtion to clean reddits
        df['cleaned_reddits'] = df['reddits'].apply(cleanRdt)
        # create two new columns called "Subjectivity" & "Polarity"
        df['subjectivity'] = df['cleaned_reddits'].apply(getSubjectivity)
        df['polarity'] = df['cleaned_reddits'].apply(getPolarity)
        df['sentiment'] = df['polarity'].apply(getSentiment)

        negative = len(df[df["sentiment"]=="negative"].cleaned_reddits)
        positive = len(df[df["sentiment"]=="positive"].cleaned_reddits)
        neutral = len(df[df["sentiment"]=="neutral"].cleaned_reddits)
        
        redsize = len(df)
        st.write ("reddit Size :", redsize)
        st.write("Positive sentiment :",positive)
        st.write("Negative sentiment :",negative)
        st.write("Neutral sentiment :",neutral)       

    #COLUMN 2    
    with col2:
        st.caption("Choose info")
        if len(coin) > 0 :            

            # See the Extracted  Raw Data : 
            if st.button("See the Extracted Raw Data"):             
                st.success("Below is the Extracted Data :")     
                st.write(raw)                                                           

            # get the polarity 
            if st.button("Get the polarity and sentiment"):
                st.success("Analysing polarity & sentiment")
                st.subheader("Analyzed Cleaned Reddit Polarity")                
                st.write(df.head(limit_scrap))

            # get the Scatter plot 
            if st.button("Get the Scatter Polarity"):
                st.set_option('deprecation.showPyplotGlobalUse', False) #ignore warning
                st.success("Analysing polarity")
                st.subheader("Analyzed Cleaned Reddit Scatter Polarity")
                
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

            # Create a Worlcloud
            if st.button("Get WordCloud for all things said about {}".format(coin)):
                st.success("Generating A WordCloud for all things said about {}".format(coin))
                text = " ".join(review for review in df.cleaned_reddits)
                stopwords = set(STOPWORDS)
                text_newALL = prepCloud(text,coin)
                wordcloud = WordCloud(stopwords=stopwords,max_words=800,max_font_size=70).generate(text_newALL)
                st.write(plt.imshow(wordcloud, interpolation='bilinear'))
                st.pyplot()
            
            #Wordcloud for Positive tweets only
            if st.button("Get WordCloud for all Positive Reddits about {}".format(coin)):
                st.success("Generating A WordCloud for all Reddits about {}".format(coin))
                if positive > 0 :
                    text_positive = " ".join(review for review in df[df["sentiment"]=="positive"].cleaned_reddits)
                    stopwords = set(STOPWORDS)
                    text_new_positive = prepCloud(text_positive,coin)
                    #text_positive=" ".join([word for word in text_positive.split() if word not in stopwords])
                    wordcloud = WordCloud(stopwords=stopwords,max_words=800,max_font_size=70).generate(text_new_positive)
                    st.write(plt.imshow(wordcloud, interpolation='bilinear'))
                    st.pyplot()   
                else :
                    st.error("There is no positive sentiment")         
            
            #Wordcloud for Negative tweets only       
            if st.button("Get WordCloud for all Negative Reddits about {}".format(coin)):
                st.success("Generating A WordCloud for all Negative Reddits about {}".format(coin))
                if negative > 0 :
                    text_negative = " ".join(review for review in df[df["sentiment"]=="negative"].cleaned_reddits)
                    stopwords = set(STOPWORDS)
                    text_new_negative = prepCloud(text_negative,coin)
                    text_negative=" ".join([word for word in text_negative.split() if word not in stopwords])
                    wordcloud = WordCloud(stopwords=stopwords,max_words=800,max_font_size=70).generate(text_new_negative)
                    st.write(plt.imshow(wordcloud, interpolation='bilinear'))
                    st.pyplot()
                else :
                    st.error("There is no negative sentiment")
    

#sidebar
st.sidebar.header("About App")
st.sidebar.info("A reddit scrapper, that scrap your input especially crypto coin/token. Extracted data will be analysed with textblob")
#tutorial
st.sidebar.header("How to use")
st.sidebar.caption ("English :")
st.sidebar.info("Type any crypto coin or something that you want to analyze, then enter how much posts do you want to analyze, and finally choose what topic do you want to analyze")
st.sidebar.caption ("Indonesia :")
st.sidebar.info("Ketik coin atau sesuatu yang ingin anda analisa, kemudian masukkan jumlah postingan yang ingin anda analisis, lalu pilih topic seperti apa yang ingin anda analisis")
st.sidebar.text("Built with Streamlit & Python3.7")
st.sidebar.text("Analyzed with textblob")

st.sidebar.header("For Any Queries/Suggestions Please reach out at :")
st.sidebar.info("hatta616@gmail.com")
st.sidebar.info("IG : @doodrobe")
st.sidebar.info("Github : @doodrobe")

if __name__ == '__main__':
    main()
