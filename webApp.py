import streamlit as st
import string 
def main():
    Menu=['Overview','home','About']
    choice=st.sidebar.selectbox('Menu',Menu)
    if choice=='Overview':
        st.title('Hate Speech for Tunisian Chat')
        st.image('https://www.theparliamentmagazine.eu/siteimg/news-main/ugc-1/fullnews/news/23340/22688_original.jpg')
        st.markdown("""
        <style>
        .big-font {
            font-size:48px !important;
        }
        </style>
        """, unsafe_allow_html=True)
        st.markdown('<p class="big-font">this App consist of <br> making a prediction for the hate speech in the Tunisian"s chat</p>', unsafe_allow_html=True)
    if choice=='home':
        st.title('Hate Speech for Tunisian Chat')
        comment=st.text_area('Enter text', height=200)
        if st.button('Predict'):
            st.write(comment)
            res = sum([i.strip(string.punctuation).isalpha() for i in comment.split()])
            if res > 120:
                st.image('https://thumbs.dreamstime.com/b/print-172932780.jpg')
            else:
                st.image('https://www.seekpng.com/png/detail/1-10353_check-mark-green-png-green-check-mark-svg.png')
if __name__=='__main__':
    main()


