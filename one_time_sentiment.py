from google_trans_new import google_translator
import streamlit as st
import pickle


with open('sentiment analysis app/pickle files/log_reg.pkl', 'rb') as f:
	model = pickle.load(f)
	
with open('sentiment analysis app/pickle files/tfidf_vectorizer.pkl', 'rb') as f:
	tfidf_vectorizer = pickle.load(f)

def one_time_sentiment():
	st.subheader('Sentiment Analyzer')
	review = st.text_area('Review Text', 'Enter your Review in Portuguese')
	
	if st.checkbox('Translate Text'):
		try:
			translator = google_translator()	
			trans_text = translator.translate(text=review, lang_tgt='en')
			st.text_area('Translated Review Text', trans_text)
		except:
			st.error('Check your Network :speak_no_evil:')
	if st.button('Get Sentiment'):
		try:
			if translator.detect(review)[0] != 'pt':
				st.warning('Review Text has to be in Portuguese language **:see_no_evil:**')
		except:
			st.warning('Check your Network :rage:')
		else:
			prediction = int(model.predict(tfidf_vectorizer.transform([review])))
			if prediction == 1:
				st.success('**Review text is Positive :joy: :white_check_mark:**')
				st.balloons()
			elif prediction == 0:
				st.error('**Review text is Negative :x: :angry:**')

if __name__ == '__main__':
	one_time_sentiment()