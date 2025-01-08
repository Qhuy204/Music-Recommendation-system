# python -m streamlit run app1.py


import streamlit as st
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('default')
import os
import tensorflow as tf
import keras
import cv2
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from spotipy.oauth2 import SpotifyClientCredentials
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity


# Load data
Music_Player = pd.read_csv("data_moods.csv")
Music_Player = Music_Player[['name','artist', 'id','mood','popularity']]
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Recommend Songs Based on Song
data_song_input = pd.read_csv("data.csv")
data_song_input = data_song_input.sample(n=50000,random_state=42).reset_index(drop=True)

numerical_features = [
    "valence", "danceability", "energy", "tempo", 
    "acousticness", "liveness", "speechiness", "instrumentalness"
]


scaler = StandardScaler()
df_scaled = pd.DataFrame(
    scaler.fit_transform(data_song_input[numerical_features]), 
    columns=numerical_features                    
)

optimal_k = 5  # Số cụm tối ưu
kmeans = KMeans(n_clusters=optimal_k, random_state=42)  # Khởi tạo mô hình K-Means
data_song_input["Cluster"] = kmeans.fit_predict(df_scaled)  # Gán nhãn cụm cho từng dòng dữ liệu

# Trực quan hóa các cụm bằng PCA (Phân tích thành phần chính)
from sklearn.decomposition import PCA

pca = PCA(n_components=2)  # Giảm dữ liệu xuống 2 chiều để trực quan hóa
pca_result = pca.fit_transform(df_scaled)  # Áp dụng PCA trên dữ liệu đã chuẩn hóa

def recommend_songs_based_on_song(song_name, df, num_recommendations=5):
    # Xác định cụm của bài hát đầu vào:
    song_cluster = df[df["name"] == song_name]["Cluster"].values[0]

    # Lọc các bài hát cùng cụm:
    same_cluster_songs = df[df["Cluster"] == song_cluster]

    # Tính độ tương đồng (cosine similarity):
    song_index = same_cluster_songs[same_cluster_songs["name"] == song_name].index[0]
    cluster_features = same_cluster_songs[numerical_features]
    similarity = cosine_similarity(cluster_features, cluster_features)

    # Lấy các bài hát tương đồng nhất:
    similar_songs = np.argsort(similarity[song_index])[-(num_recommendations + 1):-1][::-1]
    recommendations = same_cluster_songs.iloc[similar_songs][["name", "year", "artists", "popularity", "id"]]

    return recommendations



###########################
# Load pre-trained model and class names
CNN_Model = load_model("CNN_model.h5")
Emotion_Classes = ['Happy', 'Disgust', 'Fear', 'Angry', 'Neutral', 'Sad', 'Surprise']

# Making Songs Recommendations Based on Predicted Class
def Recommend_Songs(pred_class, Num_song_rcm=5):
    
    if( pred_class=='Disgust' ):

        Play = Music_Player[Music_Player['mood'] =='Sad' ]
        Play = Play.sort_values(by="popularity", ascending=False)
        Play = Play[:Num_song_rcm].reset_index(drop=True)
        return Play

    if( pred_class=='Happy' or pred_class=='Sad' ):

        Play = Music_Player[Music_Player['mood'] =='Happy' ]
        Play = Play.sort_values(by="popularity", ascending=False)
        Play = Play[:Num_song_rcm].reset_index(drop=True)
        return Play

    if( pred_class=='Fear' or pred_class=='Angry' ):

        Play = Music_Player[Music_Player['mood'] =='Calm' ]
        Play = Play.sort_values(by="popularity", ascending=False)
        Play = Play[:Num_song_rcm].reset_index(drop=True)
        return Play

    if( pred_class=='Surprise' or pred_class=='Neutral' ):

        Play = Music_Player[Music_Player['mood'] =='Energetic' ]
        Play = Play.sort_values(by="popularity", ascending=False)
        Play = Play[:Num_song_rcm].reset_index(drop=True)
        return Play

def load_and_prep_image(image: np.ndarray):
    """
    Tiền xử lý ảnh đầu vào cho model CNN.
    """
    # Chuyển sang ảnh xám
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Phát hiện khuôn mặt
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray_img, 1.1, 4)
    
    # Nếu tìm thấy khuôn mặt, cắt và xử lý vùng khuôn mặt
    if len(faces) > 0:
        x, y, w, h = faces[0]
        face_img = image[y:y + h, x:x + w]
    else:
        face_img = image  # Nếu không tìm thấy khuôn mặt, sử dụng ảnh gốc
    
    # Chuyển sang RGB và resize về kích thước 48x48
    rgb_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    rgb_img = cv2.resize(rgb_img, (48, 48))
    rgb_img = rgb_img / 255.0  # Chuẩn hóa
    return rgb_img

def predict_emotion(image: np.ndarray):
    """
    Dự đoán cảm xúc từ ảnh đầu vào.
    """
    processed_image = load_and_prep_image(image)
    img_batch = np.expand_dims(processed_image, axis=0)
    pred = CNN_Model.predict(img_batch)
    pred_class = Emotion_Classes[np.argmax(pred)]
    return pred_class, pred[0]

def display_recommendations(emotion, Num_song_rcm=5):
    # Lấy dataframe bài hát đề xuất
    recommended_songs_df = Recommend_Songs(emotion, Num_song_rcm)
    print(f"DataFrame returned: {recommended_songs_df}")  # In DataFrame ra để kiểm tra

    if recommended_songs_df is not None:
        
        # Chuyển tất cả dữ liệu thành kiểu string nếu cần
        recommended_songs_df = recommended_songs_df.applymap(str)

        # Điều chỉnh định dạng DataFrame (ví dụ: chỉ hiển thị các cột quan trọng)
        displayed_df = recommended_songs_df[['name', 'artist', 'mood', 'popularity']]

        # Hiển thị DataFrame với chiều rộng các cột tự động điều chỉnh
        st.dataframe(displayed_df, width=1200)  # Điều chỉnh chiều rộng của DataFrame
        
        # Tạo các button "Nghe" cho mỗi bài hát
        for index, row in recommended_songs_df.iterrows():
            song_id = row['name']  # Lấy ID bài hát 
            if st.button(f"Nghe {row['name']}", key=f"play_button_{index}"):
                play_song_on_spotify(song_id)

    else:
        st.error("Không có bài hát đề xuất!")

# Hàm phát nhạc trên Spotify
# def play_song_on_spotify(song_name):
#     """
#     Tìm kiếm bài hát và tạo link trực tiếp đến Spotify
#     Parameters:
#         song_name (str): Tên bài hát cần tìm
#     """
#     try:
#         # Khởi tạo Spotify client với client credentials
#         client_credentials_manager = SpotifyClientCredentials(
#             client_id="6fe0ab17a16546c1a1f9032f1b154625",
#             client_secret="bc68112c81324882bc092f8c42f8ca6d"
#         )
#         sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

#         # Tìm kiếm bài hát
#         result = sp.search(q=song_name, type="track", limit=1)

#         if result['tracks']['items']:
#             track = result['tracks']['items'][0]
#             track_id = track['id']
#             track_name = track['name']
#             artist_name = track['artists'][0]['name']

#             # Tạo direct link đến track trên Spotify
#             track_url = f"https://open.spotify.com/track/{track_id}"
            
#             # Hiển thị thông tin và link
#             st.write(f"🎵 Đã tìm thấy: **{track_name}** - {artist_name}")
#             st.markdown(f'''<a href="{track_url}" target="_blank">
#                          <button style="background-color: #1DB954; color: white; padding: 8px 16px; margin: 10px 0px;   
#                          border: none; border-radius: 20px; cursor: pointer;">
#                          ▶️ Phát trên Spotify</button></a>''', 
#                          unsafe_allow_html=True)
            
#         else:
#             st.error("❌ Không tìm thấy bài hát trên Spotify!")

#     except Exception as e:
#         st.error(f"❌ Lỗi khi tìm kiếm: {str(e)}")
        
from spotipy.oauth2 import SpotifyClientCredentials
def play_song_on_spotify(song_name):
    """
    Tìm kiếm bài hát và tạo link trực tiếp đến Spotify.
    Phát nhạc trực tiếp trên trang web (nếu có điều kiện).
    
    Parameters:
        song_name (str): Tên bài hát cần tìm
    """
    try:
        # Khởi tạo Spotify client với client credentials
        client_credentials_manager = SpotifyClientCredentials(
            client_id="6fe0ab17a16546c1a1f9032f1b154625",
            client_secret="bc68112c81324882bc092f8c42f8ca6d"
        )
        sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

        # Tìm kiếm bài hát
        result = sp.search(q=song_name, type="track", limit=1)

        if result['tracks']['items']:
            track = result['tracks']['items'][0]
            track_id = track['id']
            track_name = track['name']
            artist_name = track['artists'][0]['name']

            embedded = f'<iframe src="https://open.spotify.com/embed/track/{track_id}" width="100%" height="380" frameBorder="0" allowtransparency="true" allow="encrypted-media"></iframe>'
            # Hiển thị thông tin bài hát
            st.write(f"🎵 Đang phát: **{track_name}** - {artist_name}")

            # Nhúng Spotify player
            st.components.v1.html(
                embedded,
                height=400
            )

        else:
            st.error("❌ Không tìm thấy bài hát trên Spotify!")

    except Exception as e:
        st.error(f"❌ Lỗi khi tìm kiếm: {str(e)}")    
    
        
def show_recommendations_tab():
    # Khởi tạo session state nếu chưa có
    if 'recommended_songs_df' not in st.session_state:
        st.session_state.recommended_songs_df = None
    if 'search_performed' not in st.session_state:
        st.session_state.search_performed = False

    with st.container():
        col1, col2 = st.columns([3, 1])
        
        with col1:
            song_name = st.text_input("Nhập tên bài hát:", key="song_input")
            if st.button("Tìm kiếm", key="search_button_2"):
                # Lưu kết quả vào session state
                recommended_songs = recommend_songs_based_on_song(song_name, data_song_input, num_recommendations=Num_song_rcm)
                st.session_state.recommended_songs_df = recommended_songs.applymap(str)
                st.session_state.search_performed = True
           

    # Hiển thị kết quả nếu đã tìm kiếm
    if st.session_state.search_performed and st.session_state.recommended_songs_df is not None:
        # Container cho DataFrame
        with st.container():
            displayed_df = st.session_state.recommended_songs_df[['name', 'artists', 'year', 'popularity']]
            st.dataframe(displayed_df, width=1200)
            
            # Tạo các button "Nghe" cho mỗi bài hát
            for index, row in displayed_df.iterrows():
                song_id = row['name']  # Lấy ID bài hát 
                if st.button(f"Nghe {row['name']}", key=f"play_button_{index}"):
                    play_song_on_spotify(song_id)
        
        # # Container cho các nút "Nghe"
        # with st.container():
        #     cols = st.columns(5)  # Chia thành 5 cột để hiển thị nút
        #     for index, row in st.session_state.recommended_songs_df.iterrows():
        #         col_index = index % 5  # Xác định cột hiện tại
        #         with cols[col_index]:
        #             # Tạo một nút nhỏ gọn hơn với custom style
        #             st.markdown(f'''
        #                 <a href="https://open.spotify.com/track/{row['id']}" target="_blank">
        #                     <button style="
        #                         background-color: #1DB954;
        #                         color: white;
        #                         padding: 5px 10px;
        #                         border: none;
        #                         border-radius: 15px;
        #                         cursor: pointer;
        #                         font-size: 12px;
        #                         width: 100%;
        #                         margin: 2px 0;">
        #                         ▶️ {row['name'][:20]}{"..." if len(row['name']) > 20 else ""}
        #                     </button>
        #                 </a>
        #             ''', unsafe_allow_html=True)
        
        # try:
        #     # Khởi tạo Spotify client với client credentials
        #     client_credentials_manager = SpotifyClientCredentials(
        #         client_id="6fe0ab17a16546c1a1f9032f1b154625",
        #         client_secret="bc68112c81324882bc092f8c42f8ca6d"
        #     )
        #     sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

        #     # Tìm kiếm bài hát
        #     result = sp.search(q=song_name, type="track", limit=1)

        #     if result['tracks']['items']:
        #         track = result['tracks']['items'][0]
        #         track_id = track['id']
        #         track_name = track['name']
        #         artist_name = track['artists'][0]['name']

        #         embedded = f'<iframe src="https://open.spotify.com/embed/track/{track_id}" width="100%" height="380" frameBorder="0" allowtransparency="true" allow="encrypted-media"></iframe>'
        #         # Hiển thị thông tin bài hát
        #         st.write(f"🎵 Đang phát: **{track_name}** - {artist_name}")

        #         # Nhúng Spotify player
        #         st.components.v1.html(
        #             embedded,
        #             height=400
        #         )

        #     else:
        #         st.error("❌ Không tìm thấy bài hát trên Spotify!")

        # except Exception as e:
        #     st.error(f"❌ Lỗi khi tìm kiếm: {str(e)}")

        
# Streamlit UI
# Slider điều chỉnh số lượng bài hát đề xuất
Num_song_rcm = st.sidebar.slider(
    "Số lượng bài hát đề xuất:", 
    0, 30, 5, 1
    # Min, Max, Default, Step
)
st.title("MUSIC RECOMMENDATION SYSTEM")
tab1, tab2, tab3 = st.tabs(["Recommend based on emotion", "Recommend based on song", "Search for song"])
with tab1:
    st.write("Chụp ảnh từ camera, tải lên hoặc nói ra cảm xúc của bạn hôm nay!")
    # Chọn nguồn ảnh
    option = st.radio("Chọn nguồn:", ("Chụp từ camera", "Tải ảnh lên", "Nhập cảm xúc"))
    if option == "Chụp từ camera":
        # Chụp ảnh từ camera
        if st.button("Chụp ảnh"):
            cap = cv2.VideoCapture(0)
            ret, frame = cap.read()
            if ret:
                st.image(frame, channels="BGR", caption="Ảnh chụp từ camera")
                cap.release()
                
                # Dự đoán cảm xúc
                emotion, probabilities = predict_emotion(frame)
                st.subheader(f"Cảm xúc dự đoán: {emotion}")
                print(emotion)
                st.bar_chart(data=dict(zip(Emotion_Classes, probabilities)))
                st.write("Đề xuất bài hát:")
                display_recommendations(emotion, Num_song_rcm)
            else:
                st.error("Không thể truy cập camera!")
                cap.release()

    elif option == "Tải ảnh lên":
        # Tải ảnh từ tệp
        uploaded_file = st.file_uploader("Tải ảnh lên", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            img = Image.open(uploaded_file)
            img_np = np.array(img)
            st.image(img, caption="Ảnh đã tải lên", use_column_width=True)
            
            # Dự đoán cảm xúc
            emotion, probabilities = predict_emotion(img_np)
            st.subheader(f"Cảm xúc dự đoán: {emotion}")
            st.bar_chart(data=dict(zip(Emotion_Classes, probabilities)))
            st.write("Đề xuất bài hát:")
            display_recommendations(emotion, Num_song_rcm)
    else:
        # Nhập cảm xúc
        emotion = st.selectbox("Cảm xúc của bạn hôm nay:", Emotion_Classes, index = 0, key="emotion_select")
        st.write("Đề xuất bài hát:")
        display_recommendations(emotion, Num_song_rcm)
with tab2:
    show_recommendations_tab()
with tab3:
    st.write("Tìm kiếm bài hát và nghe trực tiếp trên Spotify!")
    name = st.text_input("Nhập tên bài hát:", key="song_search_input")
    if st.button("Tìm kiếm", key = "search_button_3"):
        play_song_on_spotify(name)
    
    

