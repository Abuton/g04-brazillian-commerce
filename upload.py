import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from wordcloud import WordCloud
import base64

@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def get_base64_of_bin_file(bin_file):
	with open(bin_file, 'rb') as f:
		data = f.read()
	return base64.b64encode(data).decode()

@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def set_png_as_page_bg(png_file):
	bin_str = get_base64_of_bin_file(png_file)
	page_bg_img = """
		<style>
			body{
			background-image: url('data:image/png;base64, %s');
			background-size: cover;
			}
		</style>
	""" % bin_str
	st.markdown(page_bg_img, unsafe_allow_html=True)
	return

def draw_wordcloud(tweets_series, sentiment):
  # word cloud visualization
  portuguese_stopwords = ['de', 'a', 'o', 'que', 'e', 'é', 'do', 'da', 'em', 'um', 'para', 'com', 'não', 
  	'uma', 'os', 'no', 'se', 'na', 'por', 'mais', 'as', 'dos', 'como', 'mas', 'ao', 'ele', 'das', 'à', 'seu',
   'sua', 'ou', 'quando', 'muito', 'nos', 'já', 'eu', 'também', 'só', 'pelo', 'pela', 'até', 'isso', 'ela',
    'entre', 'depois', 'sem', 'mesmo', 'aos', 'seus', 'quem', 'nas', 'me', 'esse', 'eles', 'você', 'essa', 'num', 
    'nem', 'suas', 'meu', 'às', 'minha', 'numa', 'pelos', 'elas', 'qual', 'nós', 'lhe', 'deles', 'essas', 'esses',
     'pelas', 'este', 'dele', 'tu', 'te', 'vocês', 'vos', 'lhes', 'meus', 'minhas', 'teu', 'tua', 'teus', 'tuas', 
     'nosso', 'nossa', 'nossos', 'nossas', 'dela', 'delas', 'esta', 'estes', 'estas', 'aquele', 'aquela', 'aqueles',
      'aquelas', 'isto', 'aquilo', 'estou', 'está', 'estamos', 'estão', 'estive', 'esteve', 'estivemos', 'estiveram', 
      'estava', 'estávamos', 'estavam', 'estivera', 'estivéramos', 'esteja', 'estejamos', 'estejam', 'estivesse', 'estivéssemos', 
      'estivessem', 'estiver', 'estivermos', 'estiverem', 'hei', 'há', 'havemos', 'hão', 'houve', 'houvemos', 'houveram', 'houvera', 
      'houvéramos', 'haja', 'hajamos', 'hajam', 'houvesse', 'houvéssemos', 'houvessem', 'houver', 'houvermos', 'houverem', 'houverei', 'houverá', 'houveremos', 'houverão', 'houveria', 'houveríamos', 'houveriam', 'sou', 'somos', 'são', 'era', 'éramos', 'eram', 'fui', 'foi', 'fomos', 'foram', 'fora', 'fôramos', 'seja', 'sejamos', 'sejam', 'fosse', 'fôssemos', 'fossem', 'for', 'formos', 'forem', 'serei', 'será', 'seremos', 'serão', 'seria', 'seríamos', 'seriam', 'tenho', 'tem', 'temos', 'tém', 'tinha', 'tínhamos', 'tinham', 'tive', 'teve', 'tivemos', 'tiveram', 'tivera', 'tivéramos', 'tenha', 'tenhamos', 'tenham', 'tivesse', 'tivéssemos', 'tivessem', 'tiver', 'tivermos', 'tiverem', 'terei', 'terá', 'teremos', 'terão', 'teria', 'teríamos', 'teriam']

  allWords = ' '.join([twts for twts in tweets_series])
  wordCloud = WordCloud(width=300, height=150, random_state=21, max_words=150, mode='RGBA',
                        max_font_size=140, stopwords=portuguese_stopwords).generate(allWords)
  plt.figure(figsize=(12, 9))
  plt.imshow(wordCloud, interpolation="bilinear")
  plt.axis('off')
  plt.tight_layout()
  plt.title(f'Most used words in a {sentiment} Review', size=30)
  plt.show()

def upload_data():
	set_png_as_page_bg('images/olist_logo.png')
	st.subheader('Use Batch Sentiment Analyzer')
	st.write('Upload the review dataset to get the sentiment of each review')

	try:
		with open('sentiment analysis app/pickle files/log_reg.pkl', 'rb') as f:
			model = pickle.load(f)
			
		with open('sentiment analysis app/pickle files/tfidf_vectorizer.pkl', 'rb') as f:
			tfidf_vectorizer = pickle.load(f)
	except:
		print('file path not specified')

	try:
		uploaded_file = st.file_uploader('Review dataset here', type='csv')
		if uploaded_file:
			reviews = pd.read_csv(uploaded_file)
			reviews = reviews.dropna(subset=['review_comment_message']).drop_duplicates(subset=['review_comment_message']).reset_index()
			st.dataframe(reviews['review_comment_message'].head(10))
			rev_msg = reviews['review_comment_message'].tolist()
			predictions = model.predict(tfidf_vectorizer.transform(rev_msg))
			# prediction_proba = model.predict_proba(tfidf_vectorizer.transform(rev_msg)[:,1])
			reviews['sentiment_class'] = pd.Series(predictions)
			# reviews['sentiment_confidence_score'] = pd.Series(prediction_proba)

			reviews['sentiment_class'] = reviews['sentiment_class'].map({1:'Positive', 0:'Negative'})
			# view_sentiment = st.button('Click to View Sentiments')
			# n = int(st.text_input('number of rows to display?'))
			st.dataframe(reviews[['review_comment_message', 'sentiment_class']].head(30))
			pct_pos = reviews['sentiment_class'].value_counts(normalize=True)*100
			st.write(pct_pos)
			pos_review = reviews[reviews['sentiment_class'] == 'Positive']['review_comment_message']
			neg_review = reviews[reviews['sentiment_class'] == 'Negative']['review_comment_message']
			# if st.checkbox('Get WordCloud analysis'):
			col1, col2 = st.beta_columns([2, 2])
			with col1:
				# col1.header('Positive Review WordCloud')
				draw_wordcloud(pos_review, 'Positive')
				col1.pyplot()
			with col2:
				# col2.header('Negative Review WordCloud')
				draw_wordcloud(neg_review, 'Negative')
				col2.pyplot()

		if reviews['sentiment_class'][0] == 'Positive':
			st.success('**Review text is Positive :joy: :yum:**')
			# st.balloons()
		elif reviews['sentiment_class'][0] == 'Negative':
			st.error('**Review text is Negative :cry: :worried:**')

	except:
		print('Upload CSV files')


if __name__ == '__main__':
	upload_data()