# An RAG Approach to Hotel Recommendation Systems Based on User Reviews

### How to use

1. Create a fork of the project and git pull. 

2. Install Python 3.10

3. Create a virtual environment and add the needed modules using the `requirements.txt` file after activating your environment.

    `pip install -r requirements.txt`

4. Create a `data` folder in the root directory of the project, with `raw`, `processed` and `embeddings` as subfolders. Your dataset would be in the `raw` subfolder.

5. On the root directory of the project, run the following command to start the program.

    `streamlit run app/streamlit_app.py`


### About the project

The project tries to use Retrieval-Augmented Generation (RAG) as a way to deliver a personalized output on what you, the user, would need while looking for a hotel based on the reviews the hotels have received as well as an explanation on how and why each hotel had been selected. It currently has the `Business` and `Tourist` traveler types to select from. 

The project is limited to using CPU instead of a GPU and can only give suggestions based on what you feed it with the dataset at that point in time. Further future works could include:

1. Having a model getting real-time data.
2. Being able to specialize further by selecting from hotels in a certain region on the map and displaying where they are located.
3. Using a GPU for the workload.
4. Having more traveler types as well as a custom traveler type saved to your account.

