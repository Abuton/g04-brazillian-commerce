from google_trans_new import google_translator
import streamlit as st
import pickle
translator = google_translator()	


with open('sentiment analysis app/pickle files/log_reg.pkl', 'rb') as f:
	model = pickle.load(f)
	
with open('sentiment analysis app/pickle files/tfidf_vectorizer.pkl', 'rb') as f:
	tfidf_vectorizer = pickle.load(f)

def one_time_sentiment():
	st.subheader('Sentiment Analyzer')
	review = st.text_area('Review Text', 'Enter your Review in Portuguese')
	trans_text = translator.translate(text=review, lang_tgt='en')
	st.write('Would you like to translate')
	if st.button('Translate'):
		st.text_area('Translated Review Text', trans_text)
	if st.button('Get Sentiment'):
		if translator.detect(review)[0] != 'pt':
			st.warning('Review Text has to be in Portuguese language **:see_no_evil:**')
		else:
			prediction = int(model.predict(tfidf_vectorizer.transform([review])))
			if prediction == 1:
				st.success('**Review text is Positive :joy: :yum:**')
				st.balloons()
			elif prediction == 0:
				st.error('**Review text is Negative :cry: :worried:**')

if __name__ == '__main__':
	one_time_sentiment()