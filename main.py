import streamlit as st
import pickle
from analysis import analysis
from upload import upload_data
from one_time_sentiment import one_time_sentiment

def main():
	"""
	A simple NLP app
	"""
	sidebar_css = """
								<style>
									.sidebar .sidebar-content{
										background-image: linear-gradient(#2e7bcf, #2e7bef);
										background-color: #011839;
										color: lightblue;
									}
								</style>
								"""
	st.markdown("""<style>
										body{
											background-color:#add8e6;
											background-image: url("images/olist_logo.png");
											color: #000052;
											}
									</style>""",
										 unsafe_allow_html=True)

	st.title('Olist User Review')
	menu = ['Data Analysis', 'Sentiment', 'Batch Sentiment', 'About']
	choice = st.sidebar.selectbox('Menu', menu)
	st.markdown(sidebar_css, unsafe_allow_html=True)

	if choice == 'About':
		st.subheader('Learn More About Sentiment Analysis')
		st.write('## Model was built using Logistic Regression :sunglasses:')
		st.write('Model was train in **Portuguese language**')
		st.write('''Reviews should be in that language (Portuguese)  \n
			Option to translate to English is Available''')
		st.write('**Meet the Team!!!**')
		# st.image('data/test.jpg', width=150, height=30)
		# st.write('Jerry\n Web App')

	if choice == 'Data Analysis':
		analysis()

	if choice == 'Sentiment':
		one_time_sentiment()

	if choice == 'Batch Sentiment':
		upload_data()


if __name__ == '__main__':
	main()