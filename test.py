import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit.components.v1 as stc
from surprise import Dataset, Reader, SVD
# load
def load_data(data):
    df = pd.read_csv(data)
    return df

def tfd_vectorizer_cosine(data):
    tdf_vec=TfidfVectorizer(stop_words='english')
    np_stranforms = tdf_vec.fit_transform(data['soup']).toarray()
    array_re = np.append(np_stranforms, data[['avg_rating_scale', 'amount_scale']], axis=1)
    cosine_sim = cosine_similarity(array_re, array_re)
    return cosine_sim

# Function that takes in course title as input and outputs most similar content course
def get_recommendations(title,df ,cosine_sim,courses,number):
    # Get the index of the course that matches the title
    idx = courses[title]

    # Get the pairwsie similarity scores of all course with that course
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar course
    sim_scores = sim_scores[1:(number+1)]

    # Get the course indices
    course_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar course
    return df[["avg_rating","title","amount","Link","image_480x270"]].iloc[course_indices]
# SVD
reader = Reader(rating_scale=(1, 5))
df_review = load_data("data_collaborative.csv")
data = Dataset.load_from_df(df_review[['user_id','course_id','rating']], reader)
svd = SVD(verbose=True, n_epochs=10)
trainset = data.build_full_trainset()
svd.fit(trainset)

def generate_recommendation_3(user_id, svd, df_review, number_of_rec, thresh=4):
    iids=df_review['course_id'].unique()
    iids_user=df_review.loc[df_review['user_id']==user_id,'course_id']
    iids_to_pred = np.setdiff1d(iids,iids_user)
    dict_re = {'uid':[],'iid':[],'est':[]}
    for iid in iids_to_pred:
        pred=svd.predict(uid = user_id, iid = iid)
        if pred.est >=thresh:
            dict_re['uid'].append(pred.uid)
            dict_re['iid'].append(pred.iid)
            dict_re['est'].append(pred.est)
    df_re = pd.DataFrame.from_dict(dict_re)
    df_re= df_re.sort_values(by='est', ascending=False).head(number_of_rec)
    # recommend 10 course
    df_recommend = df_re.merge(df_review, left_on="iid", right_on="course_id")[["avg_rating","title","amount","Link","image_480x270"]]
    return df_recommend

RESULT_TEMP = """
    <div style="width:70%;height:100%;margin:1px;padding:5px;position:relative;border-radius:5px;border-bottom-right-radius: 50px;
box-shadow:0 0 15px 5px #ccc; background-color: #a8f0c6;
  border-left: 5px solid #6c6c6c;">
<h4>{}</h4>
<p style="color:blue;"><span style="color:black;">📈Score::</span>{}</p>
<p style="color:blue;"><span style="color:black;">🔗</span><a href="{}",target="_blank">Link</a></p>
<p style="color:blue;"><span style="color:black;">💲Price:</span>{}</p>
</div>
"""
# <p style="color:blue;"><span style="color:black;">🧑‍🎓👨🏽‍🎓 Students:</span>{}</p>
def search_term_if_not_found(term,df):
	result_df = df[df['title'].str.contains(term)]
	return result_df['title']
def main():
    # reader = Reader(rating_scale=(1, 5))
    # df_review = load_data("data_contentbase.csv")
    # # print(df_review)
    # data = Dataset.load_from_df(df_review[['user_id', 'course_id','rating']], reader)
    # svd = SVD(verbose=True, n_epochs=10)
    # trainset = data.build_full_trainset()
    # svd.fit(trainset)

    st.title("Course Recommendation App")

    menu = ["Home","Recommend","About"]
    choice = st.sidebar.selectbox("Menu",menu)
    df=load_data("data_contentbase.csv")
    courses = pd.Series(df.index, index=df['title'])
    if choice == "Home":
        url="https://amber.edu.vn/wp-content/uploads/2020/01/57.jpg"
        st.image(
            url,
            width=900,  # Manually Adjust the width of the image as per requirement
        )
    elif choice == "Recommend":
        st.subheader("Recommend Courses")
        cosine_mat = tfd_vectorizer_cosine(df)
        search_term= st.text_input("Search")
        user_id = df_review[df_review['title'] == search_term]['user_id'][1]
        number_of_rec=st.sidebar.number_input("Number",3,10,5) #cairosvg
        if st.button("Recommnend"):
            if search_term is not None:
                try:
                    result = get_recommendations(search_term,df,cosine_mat,courses,number_of_rec)
                    st.subheader("Frequently Bought Together")
                    for row in result.iterrows():
                        rec_title = row[1][1]
                        rec_rating = row[1][0]
                        rec_url = row[1][3]
                        rec_price = row[1][2]
                        # st.write("Title",rec_title,)
                        stc.html(RESULT_TEMP.format(rec_title, rec_rating, rec_url, rec_price), height=200)
                        url_image = row[1][4]
                        st.image(
                            url_image,
                            width=500,  # Manually Adjust the width of the image as per requirement
                        )
                    result1 = generate_recommendation_3(user_id=user_id, svd=svd, df_review=df_review,number_of_rec=number_of_rec, thresh=4)
                    st.subheader("Students Also Bought")
                    for row in result1.iterrows():
                        rec_title = row[1][1]
                        rec_rating = row[1][0]
                        rec_url = row[1][3]
                        rec_price = row[1][2]
                        # st.write("Title",rec_title,)
                        stc.html(RESULT_TEMP.format(rec_title, rec_rating, rec_url, rec_price), height=200)
                        url_image = row[1][4]
                        st.image(
                            url_image,
                            width=500,  # Manually Adjust the width of the image as per requirement
                        )
                except:
                    result = "Not Found"
                    st.warning(result)
                    st.info("Suggested Options include")
                    result_df = search_term_if_not_found(search_term, df)
                    st.dataframe(result_df)

    else:
        st.subheader("About")
        st.text("Built with Steamlit & Pandas")
if __name__=="__main__":
    main()